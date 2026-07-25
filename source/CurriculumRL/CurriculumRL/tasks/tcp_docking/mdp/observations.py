"""阶段 D 策略观测。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ....configs.assets import ROBOT_PRIM_CONTRACT, SCENE_ENTITY_AUBO
from .runtime_state import (
    collision_clearance_enabled_mask,
    collision_group_min_clearance,
    collision_profile_one_hot,
    compute_step,
    curriculum_level_normalized,
    curriculum_stage_one_hot,
    get_runtime,
    path_mode_one_hot,
    tcp_action_mask,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def arm_joint_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene[SCENE_ENTITY_AUBO]
    joint_ids, _ = robot.find_joints(list(ROBOT_PRIM_CONTRACT.arm_joints), preserve_order=True)
    return robot.data.joint_pos[:, joint_ids]


def arm_joint_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene[SCENE_ENTITY_AUBO]
    joint_ids, _ = robot.find_joints(list(ROBOT_PRIM_CONTRACT.arm_joints), preserve_order=True)
    return robot.data.joint_vel[:, joint_ids]


def tcp_position_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    return compute_step(env)[0].position_error_b


def tcp_orientation_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    return compute_step(env)[0].orientation_error_vector_b


def tcp_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    return compute_step(env)[0].tcp_velocity_b


def target_state_one_hot(env: ManagerBasedRLEnv) -> torch.Tensor:
    indices = get_runtime(env).reset_cache.target_state_index
    return torch.nn.functional.one_hot(indices, num_classes=4).to(dtype=env.scene.env_origins.dtype)


def curriculum_level(env: ManagerBasedRLEnv) -> torch.Tensor:
    return curriculum_level_normalized(env)


def path_mode(env: ManagerBasedRLEnv) -> torch.Tensor:
    return path_mode_one_hot(env)


def collision_profile(env: ManagerBasedRLEnv) -> torch.Tensor:
    return collision_profile_one_hot(env)


def collision_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    return collision_group_min_clearance(env)


def collision_clearance_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    return collision_clearance_enabled_mask(env)


def action_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    return tcp_action_mask(env)


def curriculum_stage(env: ManagerBasedRLEnv) -> torch.Tensor:
    return curriculum_stage_one_hot(env)
