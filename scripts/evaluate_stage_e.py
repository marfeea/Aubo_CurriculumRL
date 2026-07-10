"""阶段 E 正式评估：冻结策略、关闭课程，并逐目标状态统计最终任务指标。"""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from _bootstrap import add_package_source

add_package_source()

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True, help="阶段 E PPO checkpoint 路径。")
parser.add_argument("--num-envs", type=int, default=4, help="并行环境数；必须整除每状态 episode 数。")
parser.add_argument("--episodes-per-state", type=int, default=32, help="每个目标状态、每个随机种子的完整 episode 数。")
parser.add_argument("--seeds", type=int, nargs="+", default=(7, 11, 19), help="固定评估随机种子列表。")
parser.add_argument("--json-output", type=Path, default=None, help="可选 UTF-8 JSON 结果路径。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.checkpoint.is_file():
    parser.error(f"checkpoint 不存在：{args_cli.checkpoint}")
if args_cli.num_envs <= 0 or args_cli.episodes_per_state <= 0 or args_cli.episodes_per_state % args_cli.num_envs:
    parser.error("--num-envs 和 --episodes-per-state 必须为正，且后者必须整除前者。")
if not args_cli.seeds:
    parser.error("至少提供一个 --seeds 随机种子。")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import CurriculumRL.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

from CurriculumRL.configs.curriculum import CURRICULUM_CONFIG_VERSION  # noqa: E402
from CurriculumRL.configs.task import TARGET_STATES  # noqa: E402
from CurriculumRL.tasks.tcp_docking.mdp.runtime_state import get_runtime  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def _vecnormalize_path(checkpoint: Path) -> Path:
    periodic_match = re.fullmatch(r"(.+_model)_(\d+_steps)", checkpoint.stem)
    if periodic_match:
        name = f"{periodic_match.group(1)}_vecnormalize_{periodic_match.group(2)}.pkl"
    else:
        name = f"{checkpoint.stem}_vecnormalize.pkl"
    return checkpoint.with_name(name)


def _curriculum_state_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.stem}_curriculum.json")


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


def _evaluate_target(seed: int, target_index: int, vecnormalize_path: Path) -> dict[str, object]:
    env_cfg = parse_env_cfg("CurriculumRL-TcpDocking-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = seed
    env_cfg.curriculum_enabled = False
    env_cfg.evaluation_target_state_index = target_index
    env_cfg.evaluation_curriculum_level = 4
    raw_env = gym.make("CurriculumRL-TcpDocking-v0", cfg=env_cfg)
    base = raw_env.unwrapped
    env = VecNormalize.load(vecnormalize_path, Sb3VecEnvWrapper(raw_env))
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
    env.close()
    successes = [item for item in completed if item["success"]]
    return {
        "seed": seed,
        "target_state": TARGET_STATES[target_index].name,
        "target_state_index": target_index,
        "episodes": len(completed),
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
    }


def main() -> dict[str, object]:
    vecnormalize_path = _vecnormalize_path(args_cli.checkpoint)
    curriculum_state_path = _curriculum_state_path(args_cli.checkpoint)
    if not vecnormalize_path.is_file():
        raise FileNotFoundError(f"正式评估缺少冻结的 VecNormalize 状态：{vecnormalize_path}")
    if not curriculum_state_path.is_file():
        raise FileNotFoundError(f"正式评估缺少与 checkpoint 关联的课程状态：{curriculum_state_path}")
    curriculum_snapshot = json.loads(curriculum_state_path.read_text(encoding="utf-8"))
    if curriculum_snapshot.get("config_version") != CURRICULUM_CONFIG_VERSION:
        raise ValueError("checkpoint 的课程状态版本与当前正式评估配置不一致")
    results = [
        _evaluate_target(seed, target_index, vecnormalize_path)
        for target_index in range(len(TARGET_STATES))
        for seed in args_cli.seeds
    ]
    by_target: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in results:
        by_target[str(item["target_state"])].append(item)
    summary: dict[str, object] = {
        "task": "CurriculumRL-TcpDocking-v0",
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "vecnormalize": str(vecnormalize_path.resolve()),
        "curriculum_state": str(curriculum_state_path.resolve()),
        "curriculum_config_version": CURRICULUM_CONFIG_VERSION,
        "curriculum_disabled": True,
        "evaluation_observation_curriculum_level": 4,
        "evaluation_target_distribution": "per-target fixed",
        "seeds": list(args_cli.seeds),
        "episodes_per_state_per_seed": args_cli.episodes_per_state,
        "results_by_target": dict(by_target),
    }
    print("STAGE_E_EVALUATION=" + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if args_cli.json_output is not None:
        args_cli.json_output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
