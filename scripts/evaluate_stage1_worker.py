"""P4 阶段 1 单评估单元：固定 policy × seed × z，只创建一个 SimulationApp。"""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from _bootstrap import add_package_source

add_package_source()

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", choices=("zero", "random", "ppo"), required=True, help="固定动作策略条件。")
parser.add_argument("--checkpoint", type=Path, default=None, help="PPO 条件的 checkpoint；其他条件不得传入。")
parser.add_argument("--num-envs", type=int, required=True, help="并行环境数。")
parser.add_argument("--episodes-per-mode", type=int, required=True, help="本单元完整 episode 数。")
parser.add_argument("--seed", type=int, required=True, help="固定评估随机种子。")
parser.add_argument("--path-mode-index", type=int, required=True, help="固定路径模式 z 索引。")
parser.add_argument("--json-output", type=Path, required=True, help="本单元 UTF-8 JSON 结果路径。")
parser.add_argument(
    "--behavior-trace-stride",
    type=int,
    default=4,
    help="行为轨迹每隔多少策略步保存一个样本；汇总统计始终使用全部策略步。",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs <= 0 or args_cli.episodes_per_mode <= 0 or args_cli.episodes_per_mode % args_cli.num_envs:
    parser.error("--num-envs 和 --episodes-per-mode 必须为正，且后者必须整除前者。")
if args_cli.policy == "ppo" and (args_cli.checkpoint is None or not args_cli.checkpoint.is_file()):
    parser.error("PPO 条件必须提供存在的 --checkpoint。")
if args_cli.policy != "ppo" and args_cli.checkpoint is not None:
    parser.error("零动作和随机动作条件不接受 --checkpoint。")
if args_cli.behavior_trace_stride <= 0:
    parser.error("--behavior-trace-stride 必须为正整数。")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import CurriculumRL.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

from CurriculumRL.configs.curriculum import (  # noqa: E402
    CURRICULUM_CFG,
    CURRICULUM_CONFIG_VERSION,
    CURRICULUM_V2_PATH_MODES,
)
from CurriculumRL.logic.stage_e_evaluation import (  # noqa: E402
    curriculum_state_path,
    validate_curriculum_snapshot,
    vecnormalize_path,
)
from CurriculumRL.logic.stage_p4_evaluation import (  # noqa: E402
    P4_STAGE1_EVALUATION_SCHEMA_VERSION,
    summarize_episode_behavior,
)
from CurriculumRL.tasks.tcp_docking.mdp.runtime_state import get_runtime  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


@dataclass
class EpisodeBehaviorHistory:
    raw_actions: list[list[float]] = field(default_factory=list)
    processed_actions: list[list[float]] = field(default_factory=list)
    position_error_vectors_b: list[list[float]] = field(default_factory=list)
    position_errors_m: list[float] = field(default_factory=list)
    tcp_speeds_m_s: list[float] = field(default_factory=list)
    path_reference_distances_m: list[float] = field(default_factory=list)
    path_reference_reached: list[bool] = field(default_factory=list)
    tcp_positions_before_step_b: list[list[float]] = field(default_factory=list)
    controller_protections: dict[str, list[bool]] = field(
        default_factory=lambda: {
            "singular": [],
            "delta_limited": [],
            "velocity_limited": [],
            "position_limited": [],
        }
    )


def _sample_behavior_trace(history: EpisodeBehaviorHistory, stride: int) -> list[dict[str, object]]:
    indices = list(range(0, len(history.position_errors_m), stride))
    final_index = len(history.position_errors_m) - 1
    if indices[-1] != final_index:
        indices.append(final_index)
    return [
        {
            "policy_step": index + 1,
            "raw_action": history.raw_actions[index],
            "processed_action": history.processed_actions[index],
            "tcp_position_b": history.tcp_positions_before_step_b[index],
            "position_error_vector_b": history.position_error_vectors_b[index],
            "position_error_m": history.position_errors_m[index],
            "tcp_speed_m_s": history.tcp_speeds_m_s[index],
            "path_reference_distance_m": history.path_reference_distances_m[index],
            "path_reference_reached": history.path_reference_reached[index],
            "controller_protections": {
                name: values[index] for name, values in history.controller_protections.items()
            },
        }
        for index in indices
    ]


def _termination_counts(base: object, done: np.ndarray, counts: dict[str, int]) -> None:
    if not done.any():
        return
    manager = base.termination_manager
    last_dones = manager._last_episode_dones.detach().cpu().numpy()  # noqa: SLF001 - 无公开等价统计接口。
    for index, name in enumerate(manager.active_terms):
        counts[name] += int(last_dones[done, index].sum())


def _load_ppo_environment(raw_env: object) -> tuple[object, PPO | None]:
    wrapped = Sb3VecEnvWrapper(raw_env)
    if args_cli.policy != "ppo":
        return wrapped, None
    assert args_cli.checkpoint is not None
    vecnormalize = vecnormalize_path(args_cli.checkpoint)
    curriculum_state = curriculum_state_path(args_cli.checkpoint)
    if not vecnormalize.is_file() or not curriculum_state.is_file():
        raise FileNotFoundError("P4 PPO 评估要求 checkpoint、VecNormalize 和课程状态 JSON 三件套齐全")
    validate_curriculum_snapshot(json.loads(curriculum_state.read_text(encoding="utf-8")), CURRICULUM_CONFIG_VERSION)
    environment = VecNormalize.load(vecnormalize, wrapped)
    environment.training = False
    environment.norm_reward = False
    return environment, PPO.load(args_cli.checkpoint, environment, print_system_info=False)


def _next_action(
    observation: object, agent: PPO | None, rng: np.random.Generator, action_shape: tuple[int, ...]
) -> np.ndarray:
    if args_cli.policy == "zero":
        return np.zeros(action_shape, dtype=np.float32)
    if args_cli.policy == "random":
        return rng.uniform(-1.0, 1.0, size=action_shape).astype(np.float32)
    assert agent is not None
    action, _ = agent.predict(observation, deterministic=True)
    return np.asarray(action, dtype=np.float32)


def evaluate_unit() -> dict[str, object]:
    if not 0 <= args_cli.path_mode_index < len(CURRICULUM_V2_PATH_MODES):
        raise ValueError("路径模式索引越界")
    env_cfg = parse_env_cfg("CurriculumRL-TcpDocking-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.curriculum_enabled = False
    env_cfg.evaluation_curriculum_level = 0
    env_cfg.evaluation_target_state_index = 0
    env_cfg.evaluation_path_mode_index = args_cli.path_mode_index
    env_cfg.actions.tcp_delta.diagnostics_enabled = True
    raw_env = gym.make("CurriculumRL-TcpDocking-v0", cfg=env_cfg)
    env, agent = _load_ppo_environment(raw_env)
    try:
        base = raw_env.unwrapped
        observation = env.reset()
        returns = np.zeros(args_cli.num_envs, dtype=np.float64)
        lengths = np.zeros(args_cli.num_envs, dtype=np.int64)
        trajectories: list[list[list[float]]] = [[] for _ in range(args_cli.num_envs)]
        behavior_histories = [EpisodeBehaviorHistory() for _ in range(args_cli.num_envs)]
        completed: list[dict[str, object]] = []
        causes: dict[str, int] = defaultdict(int)
        protection_steps = {"singular": 0, "delta_limited": 0, "velocity_limited": 0, "position_limited": 0}
        policy_step_count = 0
        rng = np.random.default_rng(args_cli.seed)
        while len(completed) < args_cli.episodes_per_mode:
            controller = base.action_manager.get_term("tcp_delta").controller
            tcp_position_b, _, _ = controller.current_tcp_pose_b()
            runtime = get_runtime(base)
            if runtime.metrics is None:
                raise RuntimeError("P4 评估期间未生成动作执行前的任务指标")
            metrics_before_step = runtime.metrics
            for index, point in enumerate(tcp_position_b.detach().cpu().tolist()):
                trajectories[index].append(point)
            action = _next_action(observation, agent, rng, (args_cli.num_envs, *env.action_space.shape))
            for env_index, history in enumerate(behavior_histories):
                history.raw_actions.append(action[env_index].tolist())
                history.position_error_vectors_b.append(
                    metrics_before_step.position_error_b[env_index].detach().cpu().tolist()
                )
                history.position_errors_m.append(float(metrics_before_step.distance_m[env_index].item()))
                history.tcp_speeds_m_s.append(float(metrics_before_step.tcp_speed_m_s[env_index].item()))
                history.path_reference_distances_m.append(
                    float(metrics_before_step.path_reference_distance_m[env_index].item())
                )
                history.path_reference_reached.append(
                    bool(metrics_before_step.path_reference_reached[env_index].item())
                )
                history.tcp_positions_before_step_b.append(tcp_position_b[env_index].detach().cpu().tolist())
            observation, reward, done, _ = env.step(action)
            reward = np.asarray(reward, dtype=np.float64)
            done = np.asarray(done, dtype=bool)
            if not (np.isfinite(reward).all() and np.isfinite(np.asarray(observation)).all()):
                raise RuntimeError("P4 评估期间观测或奖励出现 NaN/Inf")
            returns += reward
            lengths += 1
            policy_step_count += args_cli.num_envs
            diagnostics = controller.diagnostics
            if diagnostics is None:
                raise RuntimeError("P4 控制器保护诊断未启用")
            for name in protection_steps:
                protection_steps[name] += int(getattr(diagnostics, name).sum().item())
            _termination_counts(base, done, causes)
            runtime = get_runtime(base)
            for env_index, history in enumerate(behavior_histories):
                history.processed_actions.append(diagnostics.processed_action[env_index].detach().cpu().tolist())
                for name, values in history.controller_protections.items():
                    values.append(bool(getattr(diagnostics, name)[env_index].item()))
            for env_index in np.flatnonzero(done):
                if len(completed) >= args_cli.episodes_per_mode:
                    break
                position_error_m = float(runtime.completed_position_error_m[env_index].item())
                speed_m_s = float(runtime.completed_tcp_speed_m_s[env_index].item())
                if not np.isfinite((position_error_m, speed_m_s)).all() or len(trajectories[env_index]) < 2:
                    raise RuntimeError("未捕获完整阶段 1 终局指标或轨迹")
                history = behavior_histories[env_index]
                stage_cfg = CURRICULUM_CFG.levels[0].stage_success
                behavior_summary = summarize_episode_behavior(
                    raw_actions=history.raw_actions,
                    processed_actions=history.processed_actions,
                    tcp_positions_b=trajectories[env_index],
                    position_error_vectors_b=history.position_error_vectors_b,
                    position_errors_m=history.position_errors_m,
                    tcp_speeds_m_s=history.tcp_speeds_m_s,
                    path_reference_distances_m=history.path_reference_distances_m,
                    path_reference_reached=history.path_reference_reached,
                    controller_protections=history.controller_protections,
                    terminal_position_error_m=position_error_m,
                    terminal_tcp_speed_m_s=speed_m_s,
                    max_position_error_m=stage_cfg.max_position_error_m,
                    max_tcp_speed_m_s=stage_cfg.max_tcp_speed_m_s,
                )
                completed.append(
                    {
                        "return": float(returns[env_index]),
                        "length": int(lengths[env_index]),
                        "position_error_m": position_error_m,
                        "tcp_speed_m_s": speed_m_s,
                        "curriculum_success": bool(runtime.completed_curriculum_success[env_index].item()),
                        "formal_parking_success": bool(runtime.completed_success[env_index].item()),
                        "safety_failure": bool(runtime.completed_safety_failure[env_index].item()),
                        "timeout": bool(runtime.completed_timeout[env_index].item()),
                        "weighted_reward_components": {
                            name: float(values[env_index].item())
                            for name, values in runtime.completed_reward_component_sums.items()
                        },
                        "tcp_trajectory_b": trajectories[env_index],
                        "behavior_summary": behavior_summary,
                        "behavior_trace": _sample_behavior_trace(history, args_cli.behavior_trace_stride),
                    }
                )
            for env_index in np.flatnonzero(done):
                returns[env_index] = 0.0
                lengths[env_index] = 0
                trajectories[env_index] = []
                behavior_histories[env_index] = EpisodeBehaviorHistory()
        successes = [item for item in completed if item["curriculum_success"]]
        component_names = tuple(completed[0]["weighted_reward_components"])
        return {
            "schema_version": P4_STAGE1_EVALUATION_SCHEMA_VERSION,
            "policy": args_cli.policy,
            "checkpoint": str(args_cli.checkpoint.resolve()) if args_cli.checkpoint is not None else None,
            "seed": args_cli.seed,
            "path_mode_index": args_cli.path_mode_index,
            "path_mode": CURRICULUM_V2_PATH_MODES[args_cli.path_mode_index].name,
            "episodes": len(completed),
            "curriculum_config_version": CURRICULUM_CONFIG_VERSION,
            "curriculum_disabled": True,
            "evaluation_curriculum_level": 0,
            "evaluation_target_state_index": 0,
            "behavior_trace_stride": args_cli.behavior_trace_stride,
            "metrics": {
                "mean_return": _mean([item["return"] for item in completed]),
                "curriculum_success_rate": _mean([item["curriculum_success"] for item in completed]),
                "formal_parking_success_rate": _mean([item["formal_parking_success"] for item in completed]),
                "safety_failure_rate": _mean([item["safety_failure"] for item in completed]),
                "timeout_rate": _mean([item["timeout"] for item in completed]),
                "mean_final_position_error_m": _mean([item["position_error_m"] for item in completed]),
                "mean_minimum_position_error_m": _mean(
                    [item["behavior_summary"]["minimum_position_error_m"] for item in completed]
                ),
                "path_reference_reached_episode_rate": _mean(
                    [item["behavior_summary"]["path_reference_reached_ever"] for item in completed]
                ),
                "mean_stage_qualified_longest_run": _mean(
                    [item["behavior_summary"]["stage_qualified_longest_run"] for item in completed]
                ),
                "mean_tcp_speed_on_curriculum_success_m_s": _mean([item["tcp_speed_m_s"] for item in successes]) if successes else None,
                "mean_weighted_reward_components": {
                    name: _mean([item["weighted_reward_components"][name] for item in completed])
                    for name in component_names
                },
                "termination_counts": dict(causes),
                "controller_protection_rate_per_policy_step": {
                    name: count / policy_step_count for name, count in protection_steps.items()
                },
            },
            "episodes_detail": completed,
        }
    finally:
        env.close()


def main() -> dict[str, object]:
    report = evaluate_unit()
    args_cli.json_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args_cli.json_output.with_suffix(args_cli.json_output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args_cli.json_output)
    print("P4_STAGE1_EVALUATION_UNIT=" + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
