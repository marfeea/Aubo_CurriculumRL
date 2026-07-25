"""阶段 D 奖励分量访问器。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .runtime_state import auxiliary_reward_scales, compute_step

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _component(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    values = getattr(compute_step(env)[1], name)
    return values * auxiliary_reward_scales(env, name)


def distance_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _component(env, "distance_progress")


def best_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _component(env, "best_progress")


def proximity(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _component(env, "proximity")


def inner_docking_quality(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _component(env, "inner_docking_quality")


def low_speed_parking(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _component(env, "low_speed_parking")


def first_entry(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _component(env, "first_entry")


def tool_axis_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _component(env, "tool_axis_progress")


def path_reference_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _component(env, "path_reference_progress")


def path_reference_reached(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _component(env, "path_reference_reached")


def final_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _component(env, "final_success")


def safety_failure(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _component(env, "safety_failure")
