"""启动阶段 A 单环境静态场景并验证 articulation 运行时契约。"""

from __future__ import annotations

import argparse
import faulthandler
import time
import traceback

from _bootstrap import add_package_source

add_package_source()

_START_TIME = time.perf_counter()


def _checkpoint(message: str) -> None:
    """输出可立即看到的 A3 执行检查点。"""

    elapsed = time.perf_counter() - _START_TIME
    print(f"[A3][{elapsed:7.2f}s] {message}", flush=True)


try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError as error:
    raise SystemExit(f"未找到 Isaac Lab。请使用已安装 Isaac Lab 的 Python 运行本脚本；当前导入错误：{error}") from error

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument(
    "--traceback-timeout-s",
    type=float,
    default=60.0,
    help="步骤无输出达到该秒数时转储所有线程栈；设为 0 禁用",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

faulthandler.enable(all_threads=True)
if args_cli.traceback_timeout_s > 0.0:
    faulthandler.dump_traceback_later(args_cli.traceback_timeout_s, repeat=True)

_checkpoint(
    f"启动 AppLauncher：device={args_cli.device}, num_envs={args_cli.num_envs}, "
    f"traceback_timeout={args_cli.traceback_timeout_s}s"
)
try:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
except BaseException:
    _checkpoint("AppLauncher 启动失败，输出异常栈")
    traceback.print_exc()
    faulthandler.cancel_dump_traceback_later()
    raise
_checkpoint("AppLauncher 启动完成，开始导入 Isaac 运行模块")

import torch  # noqa: E402
from CurriculumRL.configs.assets import ROBOT_PRIM_CONTRACT  # noqa: E402
from CurriculumRL.configs.training import SIMULATION_DT_S  # noqa: E402
from CurriculumRL.runtime.scene_access import (  # noqa: E402
    apply_robot_articulation_baseline,
    validate_contact_reporting,
)
from CurriculumRL.tasks.tcp_docking.static_scene_cfg import TcpDockingStaticSceneCfg  # noqa: E402

from omni.usd import get_context  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402


def main() -> int:
    if args_cli.num_envs != 1:
        raise ValueError("阶段 A 静态 smoke 只允许 --num-envs 1")

    _checkpoint(f"创建 SimulationContext：dt={SIMULATION_DT_S}, device={args_cli.device}")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIMULATION_DT_S, device=args_cli.device))
    _checkpoint("SimulationContext 创建完成")

    _checkpoint("构造 TcpDockingStaticSceneCfg")
    scene_cfg = TcpDockingStaticSceneCfg(num_envs=1, env_spacing=4.0)
    _checkpoint("开始创建 InteractiveScene；此步骤会加载工作站、样品瓶和两台 AUBO USD")
    scene = InteractiveScene(scene_cfg)
    _checkpoint("InteractiveScene 创建完成")

    _checkpoint("获取当前 USD stage")
    stage = get_context().get_stage()
    if stage is None:
        raise RuntimeError("omni.usd 当前 stage 为空")
    _checkpoint("应用两台机器人 articulation 运行参数基线")
    articulation_paths = apply_robot_articulation_baseline(stage)
    _checkpoint(f"articulation 路径解析完成：{articulation_paths}")
    for articulation_path in articulation_paths:
        _checkpoint(f"检查接触报告：{articulation_path}")
        validate_contact_reporting(stage, articulation_path)
    _checkpoint("接触报告检查完成")

    _checkpoint("执行 sim.reset()，初始化 PhysX tensor view")
    sim.reset()
    _checkpoint("sim.reset() 完成")
    _checkpoint("执行 scene.reset()")
    scene.reset()
    _checkpoint("scene.reset() 完成")
    _checkpoint("写入场景数据并执行首个物理步")
    scene.write_data_to_sim()
    sim.step()
    scene.update(SIMULATION_DT_S)
    _checkpoint("首个物理步完成，开始验证 articulation 张量契约")

    for entity_name in ("AUBObot", "AUBObot_2"):
        _checkpoint(f"解析 {entity_name} 的关节、Flange 和 Jacobian")
        robot = scene[entity_name]
        arm_ids, arm_names = robot.find_joints(list(ROBOT_PRIM_CONTRACT.arm_joints), preserve_order=True)
        gripper_ids, gripper_names = robot.find_joints(list(ROBOT_PRIM_CONTRACT.gripper_joints), preserve_order=True)
        flange_ids, flange_names = robot.find_bodies([ROBOT_PRIM_CONTRACT.flange_body], preserve_order=True)

        _require_exact_names(entity_name, "机械臂关节", arm_names, ROBOT_PRIM_CONTRACT.arm_joints)
        _require_exact_names(entity_name, "夹爪关节", gripper_names, ROBOT_PRIM_CONTRACT.gripper_joints)
        _require_exact_names(entity_name, "末端刚体", flange_names, (ROBOT_PRIM_CONTRACT.flange_body,))
        if set(arm_ids).intersection(gripper_ids):
            raise RuntimeError(f"{entity_name}: 夹爪关节索引混入机械臂索引")

        flange_id = flange_ids[0]
        flange_state = torch.cat(
            (
                robot.data.body_pos_w[:, flange_id],
                robot.data.body_quat_w[:, flange_id],
                robot.data.body_lin_vel_w[:, flange_id],
                robot.data.body_ang_vel_w[:, flange_id],
            ),
            dim=-1,
        )
        if not torch.isfinite(flange_state).all():
            raise RuntimeError(f"{entity_name}: Flange 位姿或速度包含 NaN/Inf")

        jacobians = robot.root_physx_view.get_jacobians()
        if jacobians.ndim != 4 or jacobians.shape[0] != 1:
            raise RuntimeError(f"{entity_name}: Jacobian 形状异常：{tuple(jacobians.shape)}")
        print(
            f"{entity_name}: arm={list(arm_names)}, gripper={list(gripper_names)}, "
            f"flange_body_id={flange_id}, jacobian_shape={tuple(jacobians.shape)}",
            flush=True,
        )

    _checkpoint("释放 SimulationContext 单例和 timeline stop 回调")
    sim.clear_instance()
    _checkpoint("阶段 A 单环境静态场景 smoke 通过")
    return 0


def _require_exact_names(entity: str, label: str, actual: list[str], expected: tuple[str, ...]) -> None:
    if tuple(actual) != expected:
        raise RuntimeError(f"{entity}: {label}解析不一致，期望 {expected}，实际 {tuple(actual)}")


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        _checkpoint("A3 smoke 失败，输出完整异常栈")
        traceback.print_exc()
        raise
    finally:
        _checkpoint("关闭 SimulationApp")
        try:
            simulation_app.close()
            _checkpoint("SimulationApp 已关闭")
        finally:
            faulthandler.cancel_dump_traceback_later()
    raise SystemExit(exit_code)
