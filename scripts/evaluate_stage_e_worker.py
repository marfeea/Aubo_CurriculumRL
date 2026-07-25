"""阶段 E 单评估单元：一个子进程只创建一个 SimulationApp。"""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from _bootstrap import add_package_source

add_package_source()

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True, help="冻结的 PPO checkpoint 路径。")
parser.add_argument("--num-envs", type=int, required=True, help="并行环境数。")
parser.add_argument("--episodes-per-state", type=int, required=True, help="本单元完整 episode 数。")
parser.add_argument("--seed", type=int, required=True, help="固定评估随机种子。")
parser.add_argument("--target-state-index", type=int, required=True, help="固定目标状态索引。")
parser.add_argument("--json-output", type=Path, required=True, help="本单元 UTF-8 JSON 结果路径。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.checkpoint.is_file():
    parser.error(f"checkpoint 不存在：{args_cli.checkpoint}")
if args_cli.num_envs <= 0 or args_cli.episodes_per_state <= 0 or args_cli.episodes_per_state % args_cli.num_envs:
    parser.error("--num-envs 和 --episodes-per-state 必须为正，且后者必须整除前者。")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import CurriculumRL.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

from CurriculumRL.configs.curriculum import CURRICULUM_CONFIG_VERSION  # noqa: E402
from CurriculumRL.configs.task import TARGET_STATES  # noqa: E402
from CurriculumRL.logic.stage_e_evaluation import (  # noqa: E402
    STAGE_E_EVALUATION_SCHEMA_VERSION,
    curriculum_state_path,
    validate_curriculum_snapshot,
    vecnormalize_path,
)
from CurriculumRL.tasks.tcp_docking.mdp.runtime_state import get_runtime  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def _termination_counts(base: object, done: np.ndarray, counts: dict[str, int]) -> None:
    mask = np.asarray(done, dtype=bool)
    if not mask.any():
        return
    manager = base.termination_manager
    last_dones = manager._last_episode_dones.detach().cpu().numpy()  # noqa: SLF001 - 无公开等价统计接口。
    for index, name in enumerate(manager.active_terms):
        counts[name] += int(last_dones[mask, index].sum())


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate_unit() -> dict[str, object]:
    """完成固定 seed × target 的评估，并在进程退出前释放唯一环境。"""

    if not 0 <= args_cli.target_state_index < len(TARGET_STATES):
        raise ValueError(f"目标状态索引越界：{args_cli.target_state_index}")
    vecnormalize = vecnormalize_path(args_cli.checkpoint)
    curriculum_state = curriculum_state_path(args_cli.checkpoint)
    if not vecnormalize.is_file():
        raise FileNotFoundError(f"正式评估缺少冻结的 VecNormalize 状态：{vecnormalize}")
    if not curriculum_state.is_file():
        raise FileNotFoundError(f"正式评估缺少与 checkpoint 关联的课程状态：{curriculum_state}")
    validate_curriculum_snapshot(json.loads(curriculum_state.read_text(encoding="utf-8")), CURRICULUM_CONFIG_VERSION)

    env_cfg = parse_env_cfg("CurriculumRL-TcpDocking-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.curriculum_enabled = False
    env_cfg.evaluation_target_state_index = args_cli.target_state_index
    env_cfg.evaluation_curriculum_level = 4
    raw_env = gym.make("CurriculumRL-TcpDocking-v0", cfg=env_cfg)
    env = VecNormalize.load(vecnormalize, Sb3VecEnvWrapper(raw_env))
    try:
        base = raw_env.unwrapped
        env.training = False
        env.norm_reward = False
        agent = PPO.load(args_cli.checkpoint, env, print_system_info=False)
        observation = env.reset()
        returns = np.zeros(args_cli.num_envs, dtype=np.float64)
        lengths = np.zeros(args_cli.num_envs, dtype=np.int64)
        completed: list[dict[str, object]] = []
        causes: dict[str, int] = defaultdict(int)
        while len(completed) < args_cli.episodes_per_state:
            action, _ = agent.predict(observation, deterministic=True)
            observation, reward, done, _ = env.step(action)
            reward = np.asarray(reward, dtype=np.float64)
            done = np.asarray(done, dtype=bool)
            if not (np.isfinite(reward).all() and np.isfinite(np.asarray(observation)).all()):
                raise RuntimeError("正式评估期间观测或奖励出现 NaN/Inf")
            returns += reward
            lengths += 1
            _termination_counts(base, done, causes)
            runtime = get_runtime(base)
            for env_index in np.flatnonzero(done):
                if len(completed) >= args_cli.episodes_per_state:
                    break
                position_error_m = float(runtime.completed_position_error_m[env_index].item())
                orientation_error_rad = float(runtime.completed_orientation_error_rad[env_index].item())
                tcp_speed_m_s = float(runtime.completed_tcp_speed_m_s[env_index].item())
                if not np.isfinite((position_error_m, orientation_error_rad, tcp_speed_m_s)).all():
                    raise RuntimeError("未捕获终局指标；请检查课程 reset 与 SB3 auto-reset 的时序")
                completed.append(
                    {
                        "return": float(returns[env_index]),
                        "length": int(lengths[env_index]),
                        "position_error_m": position_error_m,
                        "orientation_error_rad": orientation_error_rad,
                        "tcp_speed_m_s": tcp_speed_m_s,
                        "success": bool(runtime.completed_success[env_index].item()),
                        "safety_failure": bool(runtime.completed_safety_failure[env_index].item()),
                        "timeout": bool(runtime.completed_timeout[env_index].item()),
                    }
                )
            returns[done] = 0.0
            lengths[done] = 0
        successes = [item for item in completed if item["success"]]
        return {
            "schema_version": STAGE_E_EVALUATION_SCHEMA_VERSION,
            "seed": args_cli.seed,
            "target_state": TARGET_STATES[args_cli.target_state_index].name,
            "target_state_index": args_cli.target_state_index,
            "episodes": len(completed),
            "curriculum_config_version": CURRICULUM_CONFIG_VERSION,
            "curriculum_disabled": True,
            "metrics": {
                "success_rate": float(np.mean([item["success"] for item in completed])),
                "mean_final_position_error_m": _mean_or_none([item["position_error_m"] for item in completed]),
                "mean_final_orientation_error_rad": _mean_or_none([item["orientation_error_rad"] for item in completed]),
                "mean_tcp_speed_on_success_m_s": _mean_or_none([item["tcp_speed_m_s"] for item in successes]),
                "illegal_collision_rate": causes["illegal_contact"] / len(completed),
                "target_disturbance_rate": causes["target_disturbed"] / len(completed),
                "safety_failure_rate": float(np.mean([item["safety_failure"] for item in completed])),
                "timeout_rate": float(np.mean([item["timeout"] for item in completed])),
                "mean_completion_time_s": float(np.mean([item["length"] for item in completed]) * base.step_dt),
                "termination_counts": dict(causes),
            },
        }
    finally:
        env.close()


def main() -> dict[str, object]:
    report = evaluate_unit()
    args_cli.json_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args_cli.json_output.with_suffix(args_cli.json_output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_output.replace(args_cli.json_output)
    print("STAGE_E_EVALUATION_UNIT=" + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
