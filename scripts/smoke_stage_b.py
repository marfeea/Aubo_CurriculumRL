"""启动阶段 B 双环境场景，验证目标部分 reset 与 ContactSensor 张量契约。"""

from __future__ import annotations

import argparse
import faulthandler
import time
import traceback

from _bootstrap import add_package_source

add_package_source()
_START_TIME = time.perf_counter()


def _checkpoint(message: str) -> None:
    print(f"[B-V2][{time.perf_counter() - _START_TIME:7.2f}s] {message}", flush=True)


from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=2)
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
from CurriculumRL.configs.assets import (  # noqa: E402
    ROBOT_PRIM_CONTRACT,
    SCENE_ENTITY_TARGET,
)
from CurriculumRL.configs.task import ILLEGAL_CONTACT_FORCE_N  # noqa: E402
from CurriculumRL.configs.training import SIMULATION_DT_S  # noqa: E402
from CurriculumRL.logic.reset_state import clone_reset_cache, create_reset_cache  # noqa: E402
from CurriculumRL.logic.terminations import illegal_contact  # noqa: E402
from CurriculumRL.runtime.scene_access import apply_robot_articulation_baseline  # noqa: E402
from CurriculumRL.tasks.tcp_docking.dynamic_scene_cfg import TcpDockingDynamicSceneCfg  # noqa: E402
from CurriculumRL.tasks.tcp_docking.events import reset_targets  # noqa: E402

from omni.usd import get_context  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402


def main() -> int:
    if args_cli.num_envs < 2:
        raise ValueError("阶段 B smoke 至少需要两个环境验证部分 reset 隔离")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIMULATION_DT_S, device=args_cli.device))
    _checkpoint("创建动态场景")
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

    target = scene[SCENE_ENTITY_TARGET]
    cache = create_reset_cache(args_cli.num_envs, device=scene.env_origins.device)
    all_env_ids = torch.arange(args_cli.num_envs, device=scene.env_origins.device)
    all_states = all_env_ids.remainder(4)
    reset_targets(scene, cache, all_env_ids, all_states)
    expected_positions = cache.target_position_w.clone()
    torch.testing.assert_close(target.data.root_pos_w, expected_positions)

    cache_before = clone_reset_cache(cache)
    target_before = target.data.root_state_w.clone()
    selected = torch.tensor([1], device=scene.env_origins.device)
    reset_targets(scene, cache, selected, torch.tensor([3], device=scene.env_origins.device))
    unselected = torch.tensor([index for index in range(args_cli.num_envs) if index != 1], device=selected.device)
    torch.testing.assert_close(target.data.root_state_w[unselected], target_before[unselected])
    for field_name in vars(cache):
        torch.testing.assert_close(
            getattr(cache, field_name)[unselected], getattr(cache_before, field_name)[unselected]
        )

    contact = scene["robot_contact"]
    forces = contact.data.net_forces_w
    if forces.ndim != 3 or forces.shape[0] != args_cli.num_envs or forces.shape[2] != 3:
        raise RuntimeError(f"ContactSensor 张量形状异常：{tuple(forces.shape)}")
    if tuple(contact.body_names).count("Base_Link") != 1:
        raise RuntimeError(f"ContactSensor 必须唯一解析 Base_Link，实际 body={contact.body_names}")
    result = illegal_contact(
        forces,
        tuple(contact.body_names),
        ROBOT_PRIM_CONTRACT.ignored_contact_bodies,
        force_threshold_n=ILLEGAL_CONTACT_FORCE_N,
    )
    if result.shape != (args_cli.num_envs,):
        raise RuntimeError(f"非法接触判定形状异常：{tuple(result.shape)}")
    print(
        f"target_state_shape={tuple(target.data.root_state_w.shape)}, contact_shape={tuple(forces.shape)}, "
        f"contact_bodies={contact.body_names}, illegal_contact={result.tolist()}",
        flush=True,
    )
    sim.clear_instance()
    _checkpoint("阶段 B 动态 smoke 通过")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        _checkpoint("阶段 B smoke 失败")
        traceback.print_exc()
        raise
    finally:
        _checkpoint("关闭 SimulationApp")
        try:
            simulation_app.close()
        finally:
            faulthandler.cancel_dump_traceback_later()
    raise SystemExit(exit_code)
