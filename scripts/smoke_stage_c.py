"""阶段 C V3：验证六维 Differential IK 方向、收敛、多环境和保护机制。"""

from __future__ import annotations

import argparse
import faulthandler
import time
import traceback
from dataclasses import dataclass

from _bootstrap import add_package_source

add_package_source()
_START_TIME = time.perf_counter()
_SIMULATION_CONTEXT = None


def _checkpoint(message: str) -> None:
    print(f"[C-V3][{time.perf_counter() - _START_TIME:7.2f}s] {message}", flush=True)


from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--traceback-timeout-s", type=float, default=60.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

faulthandler.enable(all_threads=True)
if args_cli.traceback_timeout_s > 0.0:
    faulthandler.dump_traceback_later(args_cli.traceback_timeout_s, repeat=True)

_checkpoint(f"启动 AppLauncher：device={args_cli.device}, num_envs={args_cli.num_envs}")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from CurriculumRL.configs.assets import ROBOT_PRIM_CONTRACT  # noqa: E402
from CurriculumRL.configs.differential_ik import (  # noqa: E402
    MAX_JOINT_DELTA_RAD,
    MAX_JOINT_VELOCITY_RAD_S,
    POSITION_INCREMENT_SCALE_M,
    ROTATION_INCREMENT_SCALE_RAD,
)
from CurriculumRL.configs.task import (  # noqa: E402
    TARGET_STATES,
    TARGET_TO_TOOL_ROTATION_T,
)
from CurriculumRL.configs.training import POLICY_DECIMATION, SIMULATION_DT_S  # noqa: E402
from CurriculumRL.logic.differential_ik import pose_error  # noqa: E402
from CurriculumRL.logic.reset_state import create_reset_cache  # noqa: E402
from CurriculumRL.logic.tcp_kinematics import (  # noqa: E402
    quaternion_conjugate,
    quaternion_multiply,
    world_to_root_position,
)
from CurriculumRL.runtime.scene_access import apply_robot_articulation_baseline  # noqa: E402
from CurriculumRL.tasks.tcp_docking.actions import DifferentialIKAction  # noqa: E402
from CurriculumRL.tasks.tcp_docking.dynamic_scene_cfg import TcpDockingDynamicSceneCfg  # noqa: E402
from CurriculumRL.tasks.tcp_docking.events import reset_targets  # noqa: E402

from omni.usd import get_context  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402


@dataclass(frozen=True)
class RunResult:
    initial_error_norm: torch.Tensor
    final_error_norm: torch.Tensor
    actual_delta: torch.Tensor
    protection_triggered: torch.Tensor


def main() -> int:
    global _SIMULATION_CONTEXT
    if args_cli.num_envs < 4:
        raise ValueError("阶段 C smoke 至少需要四个环境覆盖四个目标状态")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIMULATION_DT_S, device=args_cli.device))
    _SIMULATION_CONTEXT = sim
    _checkpoint(f"创建 {args_cli.num_envs} 环境场景：dt={SIMULATION_DT_S}, decimation={POLICY_DECIMATION}")
    scene = InteractiveScene(TcpDockingDynamicSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0))
    stage = get_context().get_stage()
    if stage is None:
        raise RuntimeError("omni.usd 当前 stage 为空")
    apply_robot_articulation_baseline(stage, expected_num_envs=args_cli.num_envs)
    sim.reset()
    scene.reset()
    scene.write_data_to_sim()
    sim.step()
    scene.update(SIMULATION_DT_S)

    robot = scene["AUBObot"]
    controller = DifferentialIKAction(robot, SIMULATION_DT_S, diagnostics_enabled=True)
    _validate_resolved_contract(controller, robot)

    def reset_robot() -> None:
        joint_position = robot.data.default_joint_pos.clone()
        joint_velocity = torch.zeros_like(robot.data.default_joint_vel)
        robot.write_joint_state_to_sim(joint_position, joint_velocity)
        robot.set_joint_position_target(joint_position)
        scene.write_data_to_sim()
        sim.step()
        scene.update(SIMULATION_DT_S)
        controller.reset()

    def run_increment(raw_action: torch.Tensor, physics_steps: int = POLICY_DECIMATION) -> RunResult:
        initial_position, initial_quaternion, _ = controller.current_tcp_pose_b()
        controller.process_actions(raw_action)
        initial_error = _controller_error(controller)
        protection_triggered = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=raw_action.device)
        for _ in range(physics_steps):
            controller.apply_actions()
            diagnostics = controller.diagnostics
            if diagnostics is None:
                raise RuntimeError("V3 smoke 必须启用控制器诊断")
            protection_triggered |= (
                diagnostics.singular
                | diagnostics.delta_limited
                | diagnostics.velocity_limited
                | diagnostics.position_limited
            )
            scene.write_data_to_sim()
            sim.step()
            scene.update(SIMULATION_DT_S)
            _validate_joint_state(controller, robot)
        final_position, final_quaternion, _ = controller.current_tcp_pose_b()
        final_error = _controller_error(controller)
        actual_delta = pose_error(
            initial_position,
            initial_quaternion,
            final_position,
            final_quaternion,
            position_gain=1.0,
            rotation_gain=1.0,
        )
        return RunResult(
            initial_error_norm=torch.linalg.vector_norm(initial_error, dim=-1),
            final_error_norm=torch.linalg.vector_norm(final_error, dim=-1),
            actual_delta=actual_delta,
            protection_triggered=protection_triggered,
        )

    _checkpoint("验证零动作保持")
    reset_robot()
    zero_result = run_increment(torch.zeros_like(controller.raw_action))
    if torch.any(torch.linalg.vector_norm(zero_result.actual_delta[:, :3], dim=-1) > 2.0e-3):
        raise RuntimeError(f"零动作 TCP 平移漂移过大：{zero_result.actual_delta[:, :3]}")
    if torch.any(torch.linalg.vector_norm(zero_result.actual_delta[:, 3:], dim=-1) > 1.0e-2):
        raise RuntimeError(f"零动作 TCP 旋转漂移过大：{zero_result.actual_delta[:, 3:]}")

    _checkpoint("验证六轴正负方向")
    direction_summary: dict[str, float] = {}
    axis_names = ("x", "y", "z", "rx", "ry", "rz")
    for axis, axis_name in enumerate(axis_names):
        for sign in (-1.0, 1.0):
            reset_robot()
            raw_action = torch.zeros_like(controller.raw_action)
            raw_action[:, axis] = sign
            result = run_increment(raw_action)
            signed_motion = result.actual_delta[:, axis] * sign
            minimum_motion = 1.0e-4 if axis < 3 else 1.0e-3
            if torch.any(signed_motion <= minimum_motion):
                raise RuntimeError(
                    f"{axis_name} 方向颠倒或未执行：sign={sign:+.0f}, actual={result.actual_delta[:, axis].tolist()}"
                )
            direction_summary[f"{axis_name}{sign:+.0f}"] = float(signed_motion.min().item())

    _checkpoint("验证固定位置与固定六维目标收敛")
    reset_robot()
    position_action = torch.zeros_like(controller.raw_action)
    position_action[:, :3] = position_action.new_tensor([0.8, -0.5, 0.4])
    position_result = run_increment(position_action, physics_steps=2 * POLICY_DECIMATION)
    _require_error_reduction("固定位置", position_result, ratio=0.65, max_final_error=0.006)

    reset_robot()
    six_dimensional_action = controller.raw_action.new_tensor([0.5, 0.3, -0.2, 0.4, -0.3, 0.2]).repeat(
        args_cli.num_envs, 1
    )
    pose_result = run_increment(six_dimensional_action, physics_steps=2 * POLICY_DECIMATION)
    _require_error_reduction("固定六维", pose_result, ratio=0.75, max_final_error=0.005)

    _checkpoint("验证四目标状态与多环境隔离")
    reset_robot()
    env_ids = torch.arange(args_cli.num_envs, device=controller.raw_action.device)
    state_indices = env_ids.remainder(len(TARGET_STATES))
    cache = create_reset_cache(args_cli.num_envs, device=controller.raw_action.device)
    reset_targets(scene, cache, env_ids, state_indices)
    target_position_b, target_quaternion_b = _target_tcp_poses_b(controller, scene, state_indices)
    current_position_b, current_quaternion_b, _ = controller.current_tcp_pose_b()
    full_target_error = pose_error(
        current_position_b,
        current_quaternion_b,
        target_position_b,
        target_quaternion_b,
        position_gain=1.0,
        rotation_gain=1.0,
    )
    multi_target_action = torch.cat(
        (
            full_target_error[:, :3] / POSITION_INCREMENT_SCALE_M,
            full_target_error[:, 3:] / ROTATION_INCREMENT_SCALE_RAD,
        ),
        dim=-1,
    ).clamp(-1.0, 1.0)
    initial_target_distance = torch.linalg.vector_norm(target_position_b - current_position_b, dim=-1)
    run_increment(multi_target_action, physics_steps=2 * POLICY_DECIMATION)
    final_position_b, _, _ = controller.current_tcp_pose_b()
    final_target_distance = torch.linalg.vector_norm(target_position_b - final_position_b, dim=-1)
    if torch.any(final_target_distance >= initial_target_distance):
        raise RuntimeError(
            f"四目标接近未改善：initial={initial_target_distance.tolist()}, final={final_target_distance.tolist()}"
        )

    _checkpoint("验证实际 Jacobian 上的保护机制触发")
    reset_robot()
    controller.target_tcp_position_b[:, 0] += 0.5
    controller.apply_actions()
    diagnostics = controller.diagnostics
    if diagnostics is None:
        raise RuntimeError("缺少保护机制诊断")
    protection = (
        diagnostics.singular | diagnostics.delta_limited | diagnostics.velocity_limited | diagnostics.position_limited
    )
    if not torch.all(protection):
        raise RuntimeError(f"大误差未在所有环境触发保护：{protection.tolist()}")
    if torch.any(torch.abs(diagnostics.applied_joint_delta) > MAX_JOINT_DELTA_RAD + 1.0e-7):
        raise RuntimeError("保护后关节增量仍超过单步上限")

    print(
        "V3 summary: "
        f"fixed_base={robot.is_fixed_base}, arm_joint_ids={controller.joint_ids}, "
        f"flange_body_id={controller.flange_body_id}, jacobian_body_id={controller.jacobian_body_id}, "
        f"directions={direction_summary}, "
        f"position_error={position_result.initial_error_norm.tolist()}->{position_result.final_error_norm.tolist()}, "
        f"pose_error={pose_result.initial_error_norm.tolist()}->{pose_result.final_error_norm.tolist()}, "
        f"target_distance={initial_target_distance.tolist()}->{final_target_distance.tolist()}, "
        f"protection={protection.tolist()}, sigma_min={diagnostics.minimum_singular_value.tolist()}, "
        f"damping={diagnostics.damping.tolist()}",
        flush=True,
    )
    _checkpoint("阶段 C V3 smoke 通过")
    return 0


def _controller_error(controller: DifferentialIKAction) -> torch.Tensor:
    position_b, quaternion_b, _ = controller.current_tcp_pose_b()
    return pose_error(
        position_b,
        quaternion_b,
        controller.target_tcp_position_b,
        controller.target_tcp_quaternion_b,
        position_gain=1.0,
        rotation_gain=1.0,
    )


def _require_error_reduction(label: str, result: RunResult, ratio: float, max_final_error: float) -> None:
    if torch.any(result.final_error_norm >= result.initial_error_norm * ratio):
        raise RuntimeError(
            f"{label}误差下降不足：initial={result.initial_error_norm.tolist()}, "
            f"final={result.final_error_norm.tolist()}, ratio={ratio}"
        )
    if torch.any(result.final_error_norm > max_final_error):
        raise RuntimeError(
            f"{label}未达到成功误差门限：final={result.final_error_norm.tolist()}, limit={max_final_error}"
        )


def _validate_joint_state(controller: DifferentialIKAction, robot: object) -> None:
    joint_position = robot.data.joint_pos[:, controller.joint_ids]
    joint_velocity = robot.data.joint_vel[:, controller.joint_ids]
    limits = robot.data.joint_pos_limits[:, controller.joint_ids]
    if not torch.isfinite(joint_position).all() or not torch.isfinite(joint_velocity).all():
        raise RuntimeError("机械臂关节状态包含 NaN/Inf")
    if torch.any(joint_position < limits[..., 0] - 1.0e-5) or torch.any(joint_position > limits[..., 1] + 1.0e-5):
        raise RuntimeError("机械臂实际关节位置越界")
    effective_velocity_limit = torch.minimum(
        robot.data.joint_vel_limits[:, controller.joint_ids],
        joint_velocity.new_full((), MAX_JOINT_VELOCITY_RAD_S),
    )
    if torch.any(torch.abs(joint_velocity) > effective_velocity_limit + 1.0e-3):
        raise RuntimeError(
            f"机械臂实际关节速度越界：max={torch.abs(joint_velocity).max().item()}, "
            f"limit={effective_velocity_limit.min().item()}"
        )


def _validate_resolved_contract(controller: DifferentialIKAction, robot: object) -> None:
    jacobians = robot.root_physx_view.get_jacobians()
    if len(controller.joint_ids) != 6 or set(controller.joint_ids).intersection(
        robot.find_joints(list(ROBOT_PRIM_CONTRACT.gripper_joints), preserve_order=True)[0]
    ):
        raise RuntimeError("机械臂 Jacobian 列混入夹爪关节")
    print(
        f"resolved: fixed_base={robot.is_fixed_base}, raw_jacobian_shape={tuple(jacobians.shape)}, "
        f"arm_joint_ids={controller.joint_ids}, jacobian_joint_ids={controller.jacobian_joint_ids}, "
        f"flange_body_id={controller.flange_body_id}, jacobian_body_id={controller.jacobian_body_id}",
        flush=True,
    )


def _target_tcp_poses_b(
    controller: DifferentialIKAction, scene: InteractiveScene, state_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    device = state_indices.device
    dtype = scene.env_origins.dtype
    preposition_e = torch.tensor([state.preposition_e for state in TARGET_STATES], dtype=dtype, device=device)
    target_quaternion_w = torch.tensor([state.rotation_wxyz for state in TARGET_STATES], dtype=dtype, device=device)[
        state_indices
    ]
    target_to_tool = torch.tensor(TARGET_TO_TOOL_ROTATION_T, dtype=dtype, device=device).expand(
        state_indices.shape[0], -1
    )
    target_position_w = preposition_e[state_indices] + scene.env_origins
    target_tool_quaternion_w = quaternion_multiply(target_quaternion_w, target_to_tool)
    target_position_b = world_to_root_position(
        target_position_w, controller.robot.data.root_pos_w, controller.robot.data.root_quat_w
    )
    target_quaternion_b = quaternion_multiply(
        quaternion_conjugate(controller.robot.data.root_quat_w), target_tool_quaternion_w
    )
    return target_position_b, target_quaternion_b


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        _checkpoint("阶段 C smoke 失败")
        traceback.print_exc()
        raise
    finally:
        _checkpoint("关闭 SimulationApp")
        try:
            if _SIMULATION_CONTEXT is not None:
                _SIMULATION_CONTEXT.clear_instance()
            simulation_app.close()
        finally:
            faulthandler.cancel_dump_traceback_later()
    raise SystemExit(exit_code)
