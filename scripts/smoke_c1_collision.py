"""P3 C1 故障注入：验证真实玻璃接触会触发非法接触终止。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_package_source

add_package_source()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidates", type=int, default=96, help="固定 seed 的受限关节目标个数。")
parser.add_argument("--physics-steps", type=int, default=30, help="每个目标保持的物理步数。")
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--json-output", type=Path, default=None, help="可选：保存成功注入的结构化证据。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import CurriculumRL.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from CurriculumRL.configs.assets import ROBOT_PRIM_CONTRACT, SCENE_ENTITY_AUBO  # noqa: E402
from CurriculumRL.configs.task import ILLEGAL_CONTACT_FORCE_N  # noqa: E402
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


def _force_matrix(env: object) -> torch.Tensor:
    scene = env.scene  # type: ignore[attr-defined]
    matrices = [scene[name].data.force_matrix_w for name in _C1_SENSORS]
    if any(matrix is None for matrix in matrices):
        raise RuntimeError("C1 ContactSensor 未产生 force_matrix_w")
    return torch.cat(matrices, dim=1)  # type: ignore[arg-type]


def _candidate_targets(robot: object, *, count: int, seed: int) -> torch.Tensor:
    """在实测关节限位的 70% 内固定采样，避免修改任务或执行器参数。"""

    limits = robot.data.joint_pos_limits[0, :6]  # type: ignore[attr-defined]
    center = (limits[:, 0] + limits[:, 1]) * 0.5
    half_range = (limits[:, 1] - limits[:, 0]) * 0.35
    generator = torch.Generator(device=limits.device)
    generator.manual_seed(seed)
    samples = torch.rand((count, 6), generator=generator, device=limits.device, dtype=limits.dtype)
    return center + (samples * 2.0 - 1.0) * half_range


def _check_threshold(base: object, forces: torch.Tensor) -> tuple[float, int]:
    """返回当前 C1 最大力与对应机器人刚体索引。"""

    magnitudes = torch.linalg.vector_norm(forces[0, :, 0], dim=-1)
    maximum, body_index = torch.max(magnitudes, dim=0)
    return float(maximum), int(body_index)


def main() -> int:
    if args_cli.candidates <= 0 or args_cli.physics_steps <= 0:
        raise ValueError("--candidates 与 --physics-steps 必须为正")
    env_cfg = parse_env_cfg("CurriculumRL-TcpDocking-v0", device=args_cli.device, num_envs=1)
    env_cfg.curriculum_enabled = False
    env_cfg.evaluation_target_state_index = 0
    env_cfg.evaluation_curriculum_level = 0
    env = gym.make("CurriculumRL-TcpDocking-v0", cfg=env_cfg)
    base = env.unwrapped
    best_report: dict[str, object] = {"c1_force_max_n": 0.0}
    try:
        env.reset()
        robot = base.scene[SCENE_ENTITY_AUBO]
        targets = _candidate_targets(robot, count=args_cli.candidates, seed=args_cli.seed)
        arm_ids, arm_names = robot.find_joints(list(ROBOT_PRIM_CONTRACT.arm_joints), preserve_order=True)
        if tuple(arm_names) != ROBOT_PRIM_CONTRACT.arm_joints:
            raise RuntimeError(f"机械臂关节解析异常：{arm_names}")
        for candidate_index, target in enumerate(targets):
            env.reset()
            # 仅覆盖 action term 在本次 smoke 的关节目标；每个 candidate 后完整 reset。
            zero_action = torch.zeros((1, 6), device=base.device)
            base.action_manager.process_action(zero_action)
            maximum_force = 0.0
            body_index = -1
            for physics_step in range(args_cli.physics_steps):
                base.action_manager.apply_action()
                robot.set_joint_position_target(target.unsqueeze(0), joint_ids=arm_ids)
                base.scene.write_data_to_sim()
                base.sim.step(render=False)
                base.scene.update(dt=base.physics_dt)
                forces = _force_matrix(base)
                if not torch.isfinite(forces).all():
                    raise RuntimeError("C1 force_matrix_w 出现 NaN/Inf")
                current_max, current_index = _check_threshold(base, forces)
                if current_max > maximum_force:
                    maximum_force = current_max
                    body_index = current_index
                if maximum_force > ILLEGAL_CONTACT_FORCE_N:
                    # 与常规 step 相同地由 Termination Manager 调用任务终止项。
                    base.episode_length_buf += 1
                    done = base.termination_manager.compute()
                    illegal = base.termination_manager.get_term("illegal_contact")
                    if not bool(illegal[0]) or not bool(done[0]):
                        raise RuntimeError(
                            "C1 力已超过阈值，但 illegal_contact 或总终止未触发："
                            f"force={maximum_force:.3f} N, illegal={illegal.tolist()}, done={done.tolist()}"
                        )
                    report = {
                        "candidate_index": candidate_index,
                        "physics_step": physics_step + 1,
                        "joint_target_rad": target.tolist(),
                        "body": ROBOT_PRIM_CONTRACT.contact_bodies[body_index],
                        "c1_force_max_n": maximum_force,
                        "illegal_contact": bool(illegal[0]),
                        "terminated": bool(done[0]),
                    }
                    print("C1_COLLISION_SMOKE=" + json.dumps(report, ensure_ascii=False), flush=True)
                    if args_cli.json_output is not None:
                        args_cli.json_output.parent.mkdir(parents=True, exist_ok=True)
                        args_cli.json_output.write_text(
                            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                        )
                    return 0
            if maximum_force > float(best_report["c1_force_max_n"]):
                best_report = {
                    "candidate_index": candidate_index,
                    "joint_target_rad": target.tolist(),
                    "body": ROBOT_PRIM_CONTRACT.contact_bodies[body_index] if body_index >= 0 else None,
                    "c1_force_max_n": maximum_force,
                }
        # 关节扫描可能只覆盖到工作站内部的无玻璃通道。作为明确的故障注入，
        # 仅在 smoke 中将固定基座平移到 P0 审计出的玻璃包围范围附近，不改正式场景配置。
        root_shifts_w = (
            (0.70, 0.00, 0.30),
            (-0.70, 0.00, 0.30),
            (0.00, 1.15, 0.30),
            (0.00, -1.15, 0.30),
            (0.70, 0.00, 0.75),
            (-0.70, 0.00, 0.75),
        )
        for shift_index, shift in enumerate(root_shifts_w):
            env.reset()
            root_pose = robot.data.root_state_w[:, :7].clone()
            root_pose[:, :3] += torch.tensor(shift, dtype=root_pose.dtype, device=base.device)
            robot.write_root_pose_to_sim(root_pose)
            base.action_manager.reset()
            maximum_force = 0.0
            body_index = -1
            for physics_step in range(args_cli.physics_steps):
                base.action_manager.process_action(torch.zeros((1, 6), device=base.device))
                base.action_manager.apply_action()
                base.scene.write_data_to_sim()
                base.sim.step(render=False)
                base.scene.update(dt=base.physics_dt)
                forces = _force_matrix(base)
                current_max, current_index = _check_threshold(base, forces)
                if current_max > maximum_force:
                    maximum_force = current_max
                    body_index = current_index
                if maximum_force > ILLEGAL_CONTACT_FORCE_N:
                    base.episode_length_buf += 1
                    done = base.termination_manager.compute()
                    illegal = base.termination_manager.get_term("illegal_contact")
                    if not bool(illegal[0]) or not bool(done[0]):
                        raise RuntimeError(
                            "C1 根位姿注入的力已超过阈值，但终止未触发："
                            f"force={maximum_force:.3f} N, illegal={illegal.tolist()}, done={done.tolist()}"
                        )
                    report = {
                        "root_shift_index": shift_index,
                        "root_shift_world_m": shift,
                        "physics_step": physics_step + 1,
                        "body": ROBOT_PRIM_CONTRACT.contact_bodies[body_index],
                        "c1_force_max_n": maximum_force,
                        "illegal_contact": bool(illegal[0]),
                        "terminated": bool(done[0]),
                    }
                    print("C1_COLLISION_SMOKE=" + json.dumps(report, ensure_ascii=False), flush=True)
                    if args_cli.json_output is not None:
                        args_cli.json_output.parent.mkdir(parents=True, exist_ok=True)
                        args_cli.json_output.write_text(
                            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                        )
                    return 0
            if maximum_force > float(best_report["c1_force_max_n"]):
                best_report = {
                    "root_shift_index": shift_index,
                    "root_shift_world_m": shift,
                    "body": ROBOT_PRIM_CONTRACT.contact_bodies[body_index] if body_index >= 0 else None,
                    "c1_force_max_n": maximum_force,
                }
        if args_cli.json_output is not None:
            args_cli.json_output.parent.mkdir(parents=True, exist_ok=True)
            args_cli.json_output.write_text(
                json.dumps({"status": "threshold_not_reached", **best_report}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        raise RuntimeError(
            f"在 {args_cli.candidates} 个候选、每个 {args_cli.physics_steps} 物理步内未使 C1 力超过 "
            f"{ILLEGAL_CONTACT_FORCE_N} N"
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
