"""阶段 D L0 同口径评估：零动作、随机动作或已训练 PPO 的 episode 统计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_package_source

add_package_source()

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", choices=("zero", "random", "ppo"), required=True)
parser.add_argument("--checkpoint", type=Path, help="--policy ppo 时必填的 SB3 checkpoint 路径。")
parser.add_argument("--num-envs", type=int, default=4, help="并行环境数；必须整除 --episodes。")
parser.add_argument("--episodes", type=int, default=32, help="每种策略统计的完整 episode 数。")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--json-output", type=Path, default=None, help="可选的 UTF-8 JSON 结果路径。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.episodes <= 0 or args_cli.episodes % args_cli.num_envs:
    parser.error("--episodes 必须为正数且能被 --num-envs 整除，以保证三种策略的样本数一致。")
if args_cli.policy == "ppo" and args_cli.checkpoint is None:
    parser.error("--policy ppo 必须提供 --checkpoint。")
if args_cli.checkpoint is not None and not args_cli.checkpoint.is_file():
    parser.error(f"checkpoint 不存在：{args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import CurriculumRL.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def _vecnormalize_path(checkpoint: Path) -> Path:
    """返回 train.py 对最终或周期 checkpoint 写出的配套归一化状态。"""
    import re

    periodic_match = re.fullmatch(r"(.+_model)_(\d+_steps)", checkpoint.stem)
    if periodic_match:
        name = f"{periodic_match.group(1)}_vecnormalize_{periodic_match.group(2)}.pkl"
    else:
        name = f"{checkpoint.stem}_vecnormalize.pkl"
    return checkpoint.with_name(name)


def _done_causes(base: object, done: np.ndarray, counts: dict[str, int]) -> int:
    """读取 ManagerBasedRLEnv 在本步保存的终止原因，且只计入已完成 episode。"""
    done_mask = np.asarray(done, dtype=bool)
    done_count = int(done_mask.sum())
    if done_count == 0:
        return 0
    manager = base.termination_manager
    last_dones = manager._last_episode_dones.detach().cpu().numpy()  # noqa: SLF001 - Isaac Lab 无公开等价统计接口。
    for index, name in enumerate(manager.active_terms):
        counts[name] += int(last_dones[done_mask, index].sum())
    return done_count


def main() -> dict[str, object]:
    print("[D-EVAL] 解析环境配置", flush=True)
    env_cfg = parse_env_cfg(
        "CurriculumRL-TcpDocking-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    print("[D-EVAL] 创建原始环境", flush=True)
    raw_env = gym.make("CurriculumRL-TcpDocking-v0", cfg=env_cfg)
    base = raw_env.unwrapped
    print("[D-EVAL] 包装 SB3 向量环境", flush=True)
    env = Sb3VecEnvWrapper(raw_env)
    rng = np.random.default_rng(args_cli.seed)
    agent: PPO | None = None

    if args_cli.policy == "ppo":
        assert args_cli.checkpoint is not None
        vecnormalize_path = _vecnormalize_path(args_cli.checkpoint)
        if not vecnormalize_path.is_file():
            raise FileNotFoundError(f"缺少与 checkpoint 配套的 VecNormalize 状态：{vecnormalize_path}")
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
        agent = PPO.load(args_cli.checkpoint, env, print_system_info=True)

    print("[D-EVAL] reset", flush=True)
    observation = env.reset()
    episode_returns = np.zeros(args_cli.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args_cli.num_envs, dtype=np.int64)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    causes = {name: 0 for name in base.termination_manager.active_terms}
    action_shape = (args_cli.num_envs, base.action_manager.action.shape[-1])

    print("[D-EVAL] 开始 episode 统计", flush=True)
    while len(completed_returns) < args_cli.episodes:
        if args_cli.policy == "zero":
            action = np.zeros(action_shape, dtype=np.float32)
        elif args_cli.policy == "random":
            action = rng.uniform(-1.0, 1.0, size=action_shape).astype(np.float32)
        else:
            assert agent is not None
            action, _ = agent.predict(observation, deterministic=True)

        observation, reward, done, _ = env.step(action)
        reward = np.asarray(reward, dtype=np.float64)
        done = np.asarray(done, dtype=bool)
        if not (np.isfinite(reward).all() and np.isfinite(np.asarray(observation)).all()):
            raise RuntimeError("评估期间观测或奖励出现 NaN/Inf")
        episode_returns += reward
        episode_lengths += 1
        _done_causes(base, done, causes)
        for env_index in np.flatnonzero(done):
            if len(completed_returns) >= args_cli.episodes:
                break
            completed_returns.append(float(episode_returns[env_index]))
            completed_lengths.append(int(episode_lengths[env_index]))
        episode_returns[done] = 0.0
        episode_lengths[done] = 0

    env.close()
    successes = causes.get("parking_success", 0)
    safety_failures = sum(causes.get(name, 0) for name in ("outside_workspace", "illegal_contact", "target_disturbed"))
    summary: dict[str, object] = {
        "task": "CurriculumRL-TcpDocking-v0",
        "policy": args_cli.policy,
        "seed": args_cli.seed,
        "episodes": len(completed_returns),
        "num_envs": args_cli.num_envs,
        "mean_return": float(np.mean(completed_returns)),
        "std_return": float(np.std(completed_returns)),
        "mean_length": float(np.mean(completed_lengths)),
        "successes": successes,
        "success_rate": successes / len(completed_returns),
        "safety_failures": safety_failures,
        "safety_failure_rate": safety_failures / len(completed_returns),
        "termination_counts": causes,
    }
    print("STAGE_D_EVALUATION=" + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if args_cli.json_output is not None:
        args_cli.json_output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
