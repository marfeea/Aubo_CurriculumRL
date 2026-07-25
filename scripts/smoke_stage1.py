"""P3 阶段 1：验证动作掩码、三种路径参考点和 C1 接触张量。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_package_source

add_package_source()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps-per-mode", type=int, default=64)
parser.add_argument("--output", type=Path, default=None, help="可选：输出每模式 TCP/关节轨迹 JSON。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import CurriculumRL.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from CurriculumRL.configs.assets import ROBOT_PRIM_CONTRACT, SCENE_ENTITY_AUBO  # noqa: E402
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


def _policy_observation(observation: object) -> torch.Tensor:
    if not isinstance(observation, dict) or not torch.is_tensor(observation.get("policy")):
        raise RuntimeError("策略观测必须为包含 policy 张量的字典")
    return observation["policy"]


def main() -> int:
    if args_cli.num_envs <= 0 or args_cli.steps_per_mode <= 0:
        raise ValueError("--num-envs 和 --steps-per-mode 必须为正")
    env_cfg = parse_env_cfg("CurriculumRL-TcpDocking-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    # 固定 L0 任务和目标，仅顺序切换 z；不向课程窗口提交人工 smoke 结果。
    env_cfg.curriculum_enabled = False
    env_cfg.evaluation_target_state_index = 0
    env_cfg.evaluation_curriculum_level = 0
    env = gym.make("CurriculumRL-TcpDocking-v0", cfg=env_cfg)
    base = env.unwrapped
    records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    try:
        for mode_index, mode_name in enumerate(("direct", "lateral_positive", "lateral_negative")):
            base.cfg.evaluation_path_mode_index = mode_index
            observation, _ = env.reset()
            policy = _policy_observation(observation)
            if policy.shape != (args_cli.num_envs, 52) or not torch.isfinite(policy).all():
                raise RuntimeError(f"{mode_name} 观测异常：{tuple(policy.shape)}")
            state = get_runtime(base)
            if not torch.all(state.episode_path_mode_index == mode_index):
                raise RuntimeError(f"{mode_name} 未写入当前 episode 路径模式")
            for sensor_name in _C1_SENSORS:
                matrix = base.scene[sensor_name].data.force_matrix_w
                if matrix is None or matrix.shape != (args_cli.num_envs, 1, 1, 3) or not torch.isfinite(matrix).all():
                    raise RuntimeError(f"{mode_name} 的 {sensor_name} C1 力矩阵异常：{None if matrix is None else tuple(matrix.shape)}")

            # 先发纯旋转指令，确认 L0 旋转分量在实际控制链被掩码为零。
            rotation_only = torch.zeros((args_cli.num_envs, 6), device=base.device)
            rotation_only[:, 3:] = 1.0
            env.step(rotation_only)
            processed = base.action_manager.get_term("tcp_delta").processed_actions
            if not torch.allclose(processed[:, 3:], torch.zeros_like(processed[:, 3:])):
                raise RuntimeError(f"{mode_name} 的 L0 旋转动作未被掩码")

            reached = False
            min_reference_distance = float("inf")
            for step_index in range(args_cli.steps_per_mode):
                metrics, _ = compute_step(base)
                reference = state.path_constraint_state.reference_point_b
                target = state.reset_cache.preposition_w
                robot = base.scene[SCENE_ENTITY_AUBO]
                from CurriculumRL.logic.tcp_kinematics import world_to_root_position

                target_b = world_to_root_position(target, robot.data.root_pos_w, robot.data.root_quat_w)
                destination = torch.where(
                    state.path_constraint_state.reference_reached.unsqueeze(-1), target_b, reference
                )
                action = torch.zeros((args_cli.num_envs, 6), device=base.device)
                action[:, :3] = ((destination - metrics.tcp_position_b) / 0.02).clamp(-1.0, 1.0)
                force_max = max(
                    float(base.scene[name].data.force_matrix_w.abs().max().item()) for name in _C1_SENSORS
                )
                records.append(
                    {
                        "mode": mode_name,
                        "mode_index": mode_index,
                        "step": step_index,
                        "tcp_position_b": metrics.tcp_position_b[0].tolist(),
                        "joint_position": robot.data.joint_pos[0, :6].tolist(),
                        "reference_point_b": reference[0].tolist(),
                        "reference_distance_m": float(metrics.path_reference_distance_m[0]),
                        "reference_reached": bool(metrics.path_reference_reached[0]),
                        "c1_force_max_n": force_max,
                    }
                )
                min_reference_distance = min(min_reference_distance, float(metrics.path_reference_distance_m.min()))
                observation, reward, terminated, truncated, _ = env.step(action)
                if not (torch.isfinite(_policy_observation(observation)).all() and torch.isfinite(reward).all()):
                    raise RuntimeError(f"{mode_name} 第 {step_index} 步出现 NaN/Inf")
                reached = bool(get_runtime(base).path_constraint_state.reference_reached.all())
                if reached:
                    break
            summaries.append(
                {
                    "mode": mode_name,
                    "reference_reached": reached,
                    "steps": step_index + 1,
                    "min_reference_distance_m": min_reference_distance,
                }
            )
            if not reached:
                raise RuntimeError(f"{mode_name} 在 {args_cli.steps_per_mode} 步内未达到路径参考点")
    finally:
        env.close()
    report = {"task": "CurriculumRL-TcpDocking-v0", "summaries": summaries, "trajectories": records}
    print("STAGE1_SMOKE=" + json.dumps(report, ensure_ascii=False), flush=True)
    if args_cli.output is not None:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
