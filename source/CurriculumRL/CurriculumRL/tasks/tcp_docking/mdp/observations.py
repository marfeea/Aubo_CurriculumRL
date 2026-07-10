"""阶段 D 策略观测。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ....configs.assets import ROBOT_PRIM_CONTRACT, SCENE_ENTITY_AUBO
from .runtime_state import compute_step, curriculum_level_normalized, get_runtime

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
