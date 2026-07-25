"""P5 阶段 2：验证旋转释放、姿态/位置下降、路径约束、IK 保护和 C1 张量。"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

from _bootstrap import add_package_source

add_package_source()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps-per-mode", type=int, default=96)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import CurriculumRL.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from CurriculumRL.configs.curriculum import CURRICULUM_CFG  # noqa: E402
from CurriculumRL.tasks.tcp_docking.mdp.runtime_state import compute_step, get_runtime  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


_C1_SENSORS = (
    "robot_contact_c1_base_link",
    "robot_contact_c1_link_01",
    "robot_contact_c1_link_02",
    "robot_contact_c1_link_03",
    "robot_contact_c1_link_04",
    "robot_contact_c1_link_05",
    "robot_contact_c1_flange",
)
_ROTATION_SCALE_RAD = math.radians(5.0)


def _policy_observation(observation: object) -> torch.Tensor:
    if not isinstance(observation, dict) or not torch.is_tensor(
        observation.get("policy")
    ):
        raise RuntimeError("策略观测必须为包含 policy 张量的字典")
    return observation["policy"]


def _verify_c1_sensors(base: object, mode_name: str) -> None:
    for sensor_name in _C1_SENSORS:
        matrix = base.scene[sensor_name].data.force_matrix_w
        if (
            matrix is None
            or matrix.shape != (args_cli.num_envs, 1, 1, 3)
            or not torch.isfinite(matrix).all()
        ):
            actual = None if matrix is None else tuple(matrix.shape)
            raise RuntimeError(
                f"{mode_name} 的 {sensor_name} C1 力矩阵异常：{actual}"
            )


def main() -> int:
    if args_cli.num_envs <= 0 or args_cli.steps_per_mode <= 0:
        raise ValueError("--num-envs 和 --steps-per-mode 必须为正")
    print("P5_STAGE2_SMOKE_STATUS=creating_environment", flush=True)
    env_cfg = parse_env_cfg(
        "CurriculumRL-TcpDocking-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.curriculum_enabled = False
    env_cfg.evaluation_target_state_index = 0
    env_cfg.evaluation_curriculum_level = 1
    env_cfg.actions.tcp_delta.diagnostics_enabled = True
    env = gym.make("CurriculumRL-TcpDocking-v0", cfg=env_cfg)
    base = env.unwrapped
    print("P5_STAGE2_SMOKE_STATUS=environment_ready", flush=True)
    records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    try:
        for mode_index, mode_name in enumerate(
            ("direct", "lateral_positive", "lateral_negative")
        ):
            base.cfg.evaluation_path_mode_index = mode_index
            observation, _ = env.reset()
            policy = _policy_observation(observation)
            if policy.shape != (args_cli.num_envs, 52) or not torch.isfinite(
                policy
            ).all():
                raise RuntimeError(
                    f"{mode_name} 观测异常：{tuple(policy.shape)}"
                )
            state = get_runtime(base)
            if not torch.all(state.episode_level == 1):
                raise RuntimeError(f"{mode_name} 未固定到 L1")
            if not torch.all(state.episode_path_mode_index == mode_index):
                raise RuntimeError(f"{mode_name} 未写入当前 episode 路径模式")
            expected_mask = torch.tensor(
                CURRICULUM_CFG.levels[1].tcp_action_mask,
                device=base.device,
            ).expand(args_cli.num_envs, -1)
            from CurriculumRL.tasks.tcp_docking.mdp.runtime_state import (
                tcp_action_mask,
            )

            if not torch.equal(tcp_action_mask(base), expected_mask):
                raise RuntimeError(f"{mode_name} 的 L1 六维动作掩码异常")
            _verify_c1_sensors(base, mode_name)

            rotation_probe = torch.zeros(
                (args_cli.num_envs, 6),
                device=base.device,
            )
            rotation_probe[:, 3] = 1.0
            env.step(rotation_probe)
            processed = base.action_manager.get_term(
                "tcp_delta"
            ).processed_actions
            if not torch.allclose(
                processed[:, 3],
                torch.full_like(processed[:, 3], _ROTATION_SCALE_RAD),
            ):
                raise RuntimeError(f"{mode_name} 的 L1 旋转动作未释放")

            # 清除探针带来的姿态变化，重新开始固定物理条件的受控位姿到达。
            observation, _ = env.reset()
            state = get_runtime(base)
            initial_metrics, _ = compute_step(base)
            initial_position_error = float(initial_metrics.distance_m.mean())
            initial_orientation_error = float(
                initial_metrics.orientation_error_rad.mean()
            )
            minimum_position_error = initial_position_error
            minimum_orientation_error = initial_orientation_error
            stage_success = False
            final_position_error = initial_position_error
            final_orientation_error = initial_orientation_error
            path_reference_reached = False
            rotation_probe_steps = 0
            protection_counts = {
                "singular": 0,
                "delta_limited": 0,
                "velocity_limited": 0,
                "position_limited": 0,
            }
            for step_index in range(args_cli.steps_per_mode):
                metrics, _ = compute_step(base)
                destination = torch.where(
                    state.path_constraint_state.reference_reached.unsqueeze(-1),
                    metrics.tcp_position_b,
                    state.path_constraint_state.reference_point_b,
                )
                action = torch.zeros(
                    (args_cli.num_envs, 6),
                    device=base.device,
                )
                action[:, :3] = (
                    (destination - metrics.tcp_position_b) / 0.02
                ).clamp(-1.0, 1.0)
                rotation_action = 0.25 * (
                    metrics.orientation_error_vector_b / _ROTATION_SCALE_RAD
                ).clamp(-1.0, 1.0)
                action[:, 3:] = torch.where(
                    state.path_constraint_state.reference_reached.unsqueeze(-1),
                    rotation_action,
                    torch.zeros_like(rotation_action),
                )
                if bool(
                    state.path_constraint_state.reference_reached.all()
                ):
                    rotation_probe_steps += 1
                minimum_position_error = min(
                    minimum_position_error,
                    float(metrics.distance_m.mean()),
                )
                minimum_orientation_error = min(
                    minimum_orientation_error,
                    float(metrics.orientation_error_rad.mean()),
                )
                force_max = max(
                    float(
                        base.scene[name].data.force_matrix_w.abs().max().item()
                    )
                    for name in _C1_SENSORS
                )
                records.append(
                    {
                        "mode": mode_name,
                        "mode_index": mode_index,
                        "step": step_index,
                        "tcp_position_b": metrics.tcp_position_b[0].tolist(),
                        "position_error_m": float(metrics.distance_m[0]),
                        "orientation_error_rad": float(
                            metrics.orientation_error_rad[0]
                        ),
                        "reference_reached": bool(
                            metrics.path_reference_reached[0]
                        ),
                        "c1_force_max_n": force_max,
                    }
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                if not (
                    torch.isfinite(_policy_observation(observation)).all()
                    and torch.isfinite(reward).all()
                ):
                    raise RuntimeError(
                        f"{mode_name} 第 {step_index} 步出现 NaN/Inf"
                    )
                state = get_runtime(base)
                diagnostics = base.action_manager.get_term(
                    "tcp_delta"
                ).controller.diagnostics
                if diagnostics is None:
                    raise RuntimeError("阶段 2 smoke 未启用 Differential IK 诊断")
                for name in protection_counts:
                    protection_counts[name] += int(
                        getattr(diagnostics, name).sum().item()
                    )
                done = terminated | truncated
                post_metrics, _ = compute_step(base)
                success_flags = torch.where(
                    done,
                    state.completed_curriculum_success,
                    state.curriculum_success_state.success,
                )
                stage_success = bool(success_flags.all())
                path_flags = torch.where(
                    done,
                    state.completed_curriculum_success,
                    state.path_constraint_state.reference_reached,
                )
                path_reference_reached = bool(path_flags.all())
                final_position_error = float(
                    torch.where(
                        done,
                        state.completed_position_error_m,
                        post_metrics.distance_m,
                    ).mean()
                )
                final_orientation_error = float(
                    torch.where(
                        done,
                        state.completed_orientation_error_rad,
                        post_metrics.orientation_error_rad,
                    ).mean()
                )
                if bool(done.any()) and not stage_success:
                    termination_manager = base.termination_manager
                    termination_flags = {
                        name: bool(
                            termination_manager._last_episode_dones[  # noqa: SLF001
                                :, index
                            ].any()
                        )
                        for index, name in enumerate(
                            termination_manager.active_terms
                        )
                    }
                    raise RuntimeError(
                        f"{mode_name} 在位姿成功前发生终止："
                        + json.dumps(
                            {
                                "completed_position_error_m": float(
                                    state.completed_position_error_m.mean()
                                ),
                                "completed_orientation_error_rad": float(
                                    state.completed_orientation_error_rad.mean()
                                ),
                                "completed_tcp_speed_m_s": float(
                                    state.completed_tcp_speed_m_s.mean()
                                ),
                                "completed_safety_failure": bool(
                                    state.completed_safety_failure.any()
                                ),
                                "termination_flags": termination_flags,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                if stage_success:
                    break
                if (
                    path_reference_reached
                    and rotation_probe_steps >= 2
                    and minimum_orientation_error
                    <= initial_orientation_error - 0.05
                ):
                    break

            summary = {
                "mode": mode_name,
                "stage_success": stage_success,
                "steps": step_index + 1,
                "path_reference_reached": path_reference_reached,
                "rotation_probe_steps": rotation_probe_steps,
                "initial_position_error_m": initial_position_error,
                "minimum_position_error_m": minimum_position_error,
                "final_position_error_m": final_position_error,
                "initial_orientation_error_rad": initial_orientation_error,
                "minimum_orientation_error_rad": minimum_orientation_error,
                "final_orientation_error_rad": final_orientation_error,
                "controller_protection_rate_per_policy_step": {
                    name: count
                    / ((step_index + 1) * args_cli.num_envs)
                    for name, count in protection_counts.items()
                },
            }
            summaries.append(summary)
            if not (
                path_reference_reached
                and
                minimum_position_error < initial_position_error
                and minimum_orientation_error < initial_orientation_error
            ):
                raise RuntimeError(
                    f"{mode_name} 的路径、位置或姿态 smoke 条件未满足："
                    f"{json.dumps(summary, ensure_ascii=False, sort_keys=True)}"
                )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        env.close()
    report = {
        "task": "CurriculumRL-TcpDocking-v0",
        "evaluation_curriculum_level": 1,
        "summaries": summaries,
        "trajectories": records,
    }
    print("STAGE2_SMOKE=" + json.dumps(report, ensure_ascii=False), flush=True)
    if args_cli.output is not None:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
