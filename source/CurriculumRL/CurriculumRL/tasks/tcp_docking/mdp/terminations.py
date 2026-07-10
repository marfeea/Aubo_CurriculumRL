"""阶段 D terminated 条件；超时由 Isaac Lab time_out 单独标记。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .runtime_state import compute_step

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def parking_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    return compute_step(env)[0].success


def outside_tcp_workspace(env: ManagerBasedRLEnv) -> torch.Tensor:
    return compute_step(env)[0].outside_workspace


def illegal_robot_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    return compute_step(env)[0].illegal_contact


def target_was_disturbed(env: ManagerBasedRLEnv) -> torch.Tensor:
    return compute_step(env)[0].target_disturbed
