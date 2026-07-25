"""P5 单评估单元：固定 L1/L0 × policy × seed × z，只创建一个 SimulationApp。"""
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
parser.add_argument("--policy", choices=("zero", "random", "ppo"), required=True)
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--num-envs", type=int, required=True)
parser.add_argument("--episodes-per-condition", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--path-mode-index", type=int, required=True)
parser.add_argument("--evaluation-level", type=int, choices=(0, 1), required=True)
parser.add_argument("--json-output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if (
    args_cli.num_envs <= 0
    or args_cli.episodes_per_condition <= 0
    or args_cli.episodes_per_condition % args_cli.num_envs
):
    parser.error("--num-envs 和 --episodes-per-condition 必须为正，且后者必须整除前者。")
if args_cli.policy == "ppo" and (args_cli.checkpoint is None or not args_cli.checkpoint.is_file()):
    parser.error("PPO 条件必须提供存在的 --checkpoint。")
if args_cli.policy != "ppo" and args_cli.checkpoint is not None:
    parser.error("零动作和随机动作条件不接受 --checkpoint。")

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
    CURRICULUM_V2_OBSERVATION_SCHEMA,
    CURRICULUM_V2_PATH_MODES,
)
from CurriculumRL.logic.stage_e_evaluation import (  # noqa: E402
    curriculum_state_path,
    validate_curriculum_snapshot,
    vecnormalize_path,
)
from CurriculumRL.logic.stage_p5_evaluation import P5_STAGE2_EVALUATION_SCHEMA_VERSION  # noqa: E402
from CurriculumRL.tasks.tcp_docking.mdp.runtime_state import compute_step, get_runtime  # noqa: E402
from isaaclab_rl.sb3 import Sb3VecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


_PROTECTION_NAMES = ("singular", "delta_limited", "velocity_limited", "position_limited")


def _mean(values: list[float | bool]) -> float:
    return float(np.mean(values))


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile))


def _termination_counts(base: object, done: np.ndarray, counts: dict[str, int]) -> None:
    if not done.any():
        return
    manager = base.termination_manager
    last_dones = manager._last_episode_dones.detach().cpu().numpy()  # noqa: SLF001
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
        raise FileNotFoundError("P5 PPO 评估要求 checkpoint、VecNormalize 和课程状态 JSON 三件套齐全")
    validate_curriculum_snapshot(
        json.loads(curriculum_state.read_text(encoding="utf-8")),
        CURRICULUM_CONFIG_VERSION,
    )
    environment = VecNormalize.load(vecnormalize, wrapped)
    environment.training = False
    environment.norm_reward = False
    return environment, PPO.load(args_cli.checkpoint, environment, print_system_info=False)


def _next_action(
    observation: object,
    agent: PPO | None,
    rng: np.random.Generator,
    action_shape: tuple[int, ...],
) -> np.ndarray:
    if args_cli.policy == "zero":
        return np.zeros(action_shape, dtype=np.float32)
    if args_cli.policy == "random":
        return rng.uniform(-1.0, 1.0, size=action_shape).astype(np.float32)
    assert agent is not None
    action, _ = agent.predict(observation, deterministic=True)
    return np.asarray(action, dtype=np.float32)


def _joint_limit_violation(controller: object) -> float:
    robot = controller.robot
    positions = robot.data.joint_pos[:, controller.joint_ids]
    limits = robot.data.joint_pos_limits[:, controller.joint_ids]
    violation = np.maximum(
        (limits[..., 0] - positions).clamp_min(0.0).detach().cpu().numpy(),
        (positions - limits[..., 1]).clamp_min(0.0).detach().cpu().numpy(),
    )
    return float(violation.max(initial=0.0))


def evaluate_unit() -> dict[str, object]:
    if not 0 <= args_cli.path_mode_index < len(CURRICULUM_V2_PATH_MODES):
        raise ValueError("路径模式索引越界")
    env_cfg = parse_env_cfg(
        "CurriculumRL-TcpDocking-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.curriculum_enabled = False
    env_cfg.evaluation_curriculum_level = args_cli.evaluation_level
    env_cfg.evaluation_target_state_index = 0
    env_cfg.evaluation_path_mode_index = args_cli.path_mode_index
    env_cfg.actions.tcp_delta.diagnostics_enabled = True
    raw_env = gym.make("CurriculumRL-TcpDocking-v0", cfg=env_cfg)
    env, agent = _load_ppo_environment(raw_env)
    try:
        base = raw_env.unwrapped
        observation = env.reset()
        runtime = get_runtime(base)
        expected_mask = np.asarray(
            CURRICULUM_CFG.levels[args_cli.evaluation_level].tcp_action_mask,
            dtype=np.float32,
        )
        action_mask_offset = 0
        for field in CURRICULUM_V2_OBSERVATION_SCHEMA:
            if field.name == "tcp_action_mask":
                break
            action_mask_offset += field.dimension
        else:
            raise RuntimeError("固定观测 schema 缺少 tcp_action_mask")
        actual_mask = (
            base.observation_manager.compute_group("policy")[
                :,
                action_mask_offset : action_mask_offset + expected_mask.size,
            ]
            .detach()
            .cpu()
            .numpy()
        )
        if not np.allclose(actual_mask, expected_mask[None, :]):
            raise RuntimeError("P5 固定评估的动作掩码观测与等级配置不一致")

        returns = np.zeros(args_cli.num_envs, dtype=np.float64)
        lengths = np.zeros(args_cli.num_envs, dtype=np.int64)
        initial_position = np.full(args_cli.num_envs, np.nan, dtype=np.float64)
        initial_orientation = np.full(args_cli.num_envs, np.nan, dtype=np.float64)
        trajectories: list[list[list[float]]] = [[] for _ in range(args_cli.num_envs)]
        completed: list[dict[str, object]] = []
        causes: dict[str, int] = defaultdict(int)
        protection_steps = {name: 0 for name in _PROTECTION_NAMES}
        protection_streak = {
            name: np.zeros(args_cli.num_envs, dtype=np.int64) for name in _PROTECTION_NAMES
        }
        protection_max_streak = {name: 0 for name in _PROTECTION_NAMES}
        action_saturation_counts = {"translation": 0, "rotation": 0}
        policy_step_count = 0
        max_joint_limit_violation_rad = 0.0
        rng = np.random.default_rng(args_cli.seed)

        while len(completed) < args_cli.episodes_per_condition:
            metrics, _ = compute_step(base)
            missing_initial = ~np.isfinite(initial_position)
            initial_position[missing_initial] = metrics.distance_m.detach().cpu().numpy()[missing_initial]
            initial_orientation[missing_initial] = metrics.orientation_error_rad.detach().cpu().numpy()[
                missing_initial
            ]
            for index, point in enumerate(metrics.tcp_position_b.detach().cpu().tolist()):
                trajectories[index].append(point)

            action = _next_action(
                observation,
                agent,
                rng,
                (args_cli.num_envs, *env.action_space.shape),
            )
            action_saturation_counts["translation"] += int(
                (np.abs(action[:, :3]) >= 0.999).sum()
            )
            action_saturation_counts["rotation"] += int(
                (np.abs(action[:, 3:]) >= 0.999).sum()
            )
            observation, reward, done, _ = env.step(action)
            reward = np.asarray(reward, dtype=np.float64)
            done = np.asarray(done, dtype=bool)
            if not (
                np.isfinite(reward).all()
                and np.isfinite(np.asarray(observation)).all()
            ):
                raise RuntimeError("P5 评估期间观测或奖励出现 NaN/Inf")
            returns += reward
            lengths += 1
            policy_step_count += args_cli.num_envs

            controller = base.action_manager.get_term("tcp_delta").controller
            diagnostics = controller.diagnostics
            if diagnostics is None:
                raise RuntimeError("P5 控制器保护诊断未启用")
            for name in _PROTECTION_NAMES:
                flags = getattr(diagnostics, name).detach().cpu().numpy().astype(bool)
                protection_steps[name] += int(flags.sum())
                protection_streak[name] = np.where(flags, protection_streak[name] + 1, 0)
                protection_max_streak[name] = max(
                    protection_max_streak[name],
                    int(protection_streak[name].max(initial=0)),
                )
            max_joint_limit_violation_rad = max(
                max_joint_limit_violation_rad,
                _joint_limit_violation(controller),
            )
            _termination_counts(base, done, causes)
            runtime = get_runtime(base)
            for env_index in np.flatnonzero(done):
                if len(completed) >= args_cli.episodes_per_condition:
                    break
                final_position = float(runtime.completed_position_error_m[env_index].item())
                final_orientation = float(runtime.completed_orientation_error_rad[env_index].item())
                final_speed = float(runtime.completed_tcp_speed_m_s[env_index].item())
                finite_values = (
                    initial_position[env_index],
                    initial_orientation[env_index],
                    final_position,
                    final_orientation,
                    final_speed,
                )
                if not np.isfinite(finite_values).all() or len(trajectories[env_index]) < 2:
                    raise RuntimeError("未捕获完整 P5 终局指标或轨迹")
                completed.append(
                    {
                        "return": float(returns[env_index]),
                        "length": int(lengths[env_index]),
                        "initial_position_error_m": float(initial_position[env_index]),
                        "final_position_error_m": final_position,
                        "position_error_improvement_m": float(
                            initial_position[env_index] - final_position
                        ),
                        "initial_orientation_error_rad": float(initial_orientation[env_index]),
                        "final_orientation_error_rad": final_orientation,
                        "orientation_error_improvement_rad": float(
                            initial_orientation[env_index] - final_orientation
                        ),
                        "tcp_speed_m_s": final_speed,
                        "curriculum_success": bool(
                            runtime.completed_curriculum_success[env_index].item()
                        ),
                        "formal_parking_success": bool(
                            runtime.completed_success[env_index].item()
                        ),
                        "safety_failure": bool(
                            runtime.completed_safety_failure[env_index].item()
                        ),
                        "timeout": bool(runtime.completed_timeout[env_index].item()),
                        "weighted_reward_components": {
                            name: float(values[env_index].item())
                            for name, values in runtime.completed_reward_component_sums.items()
                        },
                        "tcp_trajectory_b": trajectories[env_index],
                    }
                )
            for env_index in np.flatnonzero(done):
                returns[env_index] = 0.0
                lengths[env_index] = 0
                initial_position[env_index] = np.nan
                initial_orientation[env_index] = np.nan
                trajectories[env_index] = []
                for name in _PROTECTION_NAMES:
                    protection_streak[name][env_index] = 0

        successes = [item for item in completed if item["curriculum_success"]]
        component_names = tuple(completed[0]["weighted_reward_components"])
        stage_cfg = CURRICULUM_CFG.levels[args_cli.evaluation_level].stage_success
        component_denominator = policy_step_count * 3
        return {
            "schema_version": P5_STAGE2_EVALUATION_SCHEMA_VERSION,
            "policy": args_cli.policy,
            "checkpoint": (
                str(args_cli.checkpoint.resolve())
                if args_cli.checkpoint is not None
                else None
            ),
            "seed": args_cli.seed,
            "path_mode_index": args_cli.path_mode_index,
            "path_mode": CURRICULUM_V2_PATH_MODES[args_cli.path_mode_index].name,
            "episodes": len(completed),
            "curriculum_config_version": CURRICULUM_CONFIG_VERSION,
            "curriculum_disabled": True,
            "evaluation_curriculum_level": args_cli.evaluation_level,
            "evaluation_target_state_index": 0,
            "stage_success_thresholds": {
                "max_position_error_m": stage_cfg.max_position_error_m,
                "max_orientation_error_rad": stage_cfg.max_orientation_error_rad,
                "max_tcp_speed_m_s": stage_cfg.max_tcp_speed_m_s,
                "required_dwell_steps": stage_cfg.required_dwell_steps,
                "require_path_reference_reached": stage_cfg.require_path_reference_reached,
            },
            "metrics": {
                "mean_return": _mean([item["return"] for item in completed]),
                "curriculum_success_rate": _mean(
                    [item["curriculum_success"] for item in completed]
                ),
                "formal_parking_success_rate": _mean(
                    [item["formal_parking_success"] for item in completed]
                ),
                "safety_failure_rate": _mean(
                    [item["safety_failure"] for item in completed]
                ),
                "timeout_rate": _mean([item["timeout"] for item in completed]),
                "mean_final_position_error_m": _mean(
                    [item["final_position_error_m"] for item in completed]
                ),
                "p95_final_position_error_m": _percentile(
                    [item["final_position_error_m"] for item in completed],
                    95.0,
                ),
                "mean_position_error_improvement_m": _mean(
                    [item["position_error_improvement_m"] for item in completed]
                ),
                "mean_final_orientation_error_rad": _mean(
                    [item["final_orientation_error_rad"] for item in completed]
                ),
                "p95_final_orientation_error_rad": _percentile(
                    [item["final_orientation_error_rad"] for item in completed],
                    95.0,
                ),
                "mean_orientation_error_improvement_rad": _mean(
                    [item["orientation_error_improvement_rad"] for item in completed]
                ),
                "mean_tcp_speed_on_curriculum_success_m_s": (
                    _mean([item["tcp_speed_m_s"] for item in successes])
                    if successes
                    else None
                ),
                "mean_weighted_reward_components": {
                    name: _mean(
                        [
                            item["weighted_reward_components"][name]
                            for item in completed
                        ]
                    )
                    for name in component_names
                },
                "termination_counts": dict(causes),
                "controller_protection_rate_per_policy_step": {
                    name: count / policy_step_count
                    for name, count in protection_steps.items()
                },
                "controller_protection_max_consecutive_policy_steps": protection_max_streak,
                "action_component_saturation_rate": {
                    name: count / component_denominator
                    for name, count in action_saturation_counts.items()
                },
                "max_joint_position_limit_violation_rad": max_joint_limit_violation_rad,
            },
            "episodes_detail": completed,
        }
    finally:
        env.close()


def main() -> dict[str, object]:
    report = evaluate_unit()
    args_cli.json_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args_cli.json_output.with_suffix(
        args_cli.json_output.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args_cli.json_output)
    print(
        "P5_STAGE2_EVALUATION_UNIT="
        + json.dumps(report, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return report


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
