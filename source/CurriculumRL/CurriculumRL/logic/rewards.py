"""不依赖 Isaac 的阶段 D 奖励状态更新。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RewardState:
    previous_distance_m: torch.Tensor
    best_distance_m: torch.Tensor
    previous_axis_alignment: torch.Tensor
    has_entered: torch.Tensor


@dataclass(frozen=True)
class RewardComponents:
    distance_progress: torch.Tensor
    best_progress: torch.Tensor
    proximity: torch.Tensor
    inner_docking_quality: torch.Tensor
    low_speed_parking: torch.Tensor
    first_entry: torch.Tensor
    tool_axis_progress: torch.Tensor
    final_success: torch.Tensor
    safety_failure: torch.Tensor


def create_reward_state(num_envs: int, *, device: torch.device | str = "cpu") -> RewardState:
    return RewardState(
        previous_distance_m=torch.full((num_envs,), torch.nan, device=device),
        best_distance_m=torch.full((num_envs,), torch.inf, device=device),
        previous_axis_alignment=torch.full((num_envs,), torch.nan, device=device),
        has_entered=torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def reset_reward_state(state: RewardState, env_ids: torch.Tensor) -> None:
    state.previous_distance_m[env_ids] = torch.nan
    state.best_distance_m[env_ids] = torch.inf
    state.previous_axis_alignment[env_ids] = torch.nan
    state.has_entered[env_ids] = False


def update_reward_state(
    state: RewardState,
    distance_m: torch.Tensor,
    orientation_error_rad: torch.Tensor,
    tcp_speed_m_s: torch.Tensor,
    axis_alignment: torch.Tensor,
    parking_inside: torch.Tensor,
    success: torch.Tensor,
    safety_failure: torch.Tensor,
    *,
    proximity_length_scale_m: float,
    orientation_quality_scale_rad: float,
    speed_quality_scale_m_s: float,
    update_mask: torch.Tensor | None = None,
) -> RewardComponents:
    """计算一次策略步奖励分量并提交历史；首次样本不产生伪进展。"""

    if min(proximity_length_scale_m, orientation_quality_scale_rad, speed_quality_scale_m_s) <= 0.0:
        raise ValueError("奖励质量尺度必须为正数")
    if update_mask is None:
        update_mask = torch.ones_like(distance_m, dtype=torch.bool)
    previous_valid = torch.isfinite(state.previous_distance_m) & update_mask
    best_valid = torch.isfinite(state.best_distance_m)
    axis_valid = torch.isfinite(state.previous_axis_alignment) & update_mask
    distance_progress = torch.where(previous_valid, state.previous_distance_m - distance_m, 0.0)
    best_progress = torch.where(best_valid, (state.best_distance_m - distance_m).clamp_min(0.0), 0.0)
    tool_axis_progress = torch.where(axis_valid, (axis_alignment - state.previous_axis_alignment).clamp_min(0.0), 0.0)
    proximity = torch.exp(-distance_m / proximity_length_scale_m)
    orientation_quality = torch.exp(-orientation_error_rad / orientation_quality_scale_rad)
    speed_quality = torch.exp(-tcp_speed_m_s / speed_quality_scale_m_s)
    first_entry = parking_inside & ~state.has_entered & update_mask

    state.previous_distance_m[update_mask] = distance_m[update_mask]
    state.best_distance_m[update_mask] = torch.minimum(state.best_distance_m[update_mask], distance_m[update_mask])
    state.previous_axis_alignment[update_mask] = axis_alignment[update_mask]
    state.has_entered[update_mask] |= parking_inside[update_mask]

    return RewardComponents(
        distance_progress=distance_progress,
        best_progress=best_progress,
        proximity=proximity * update_mask,
        inner_docking_quality=parking_inside * update_mask * 0.5 * (orientation_quality + speed_quality),
        low_speed_parking=parking_inside * update_mask * speed_quality,
        first_entry=first_entry.to(distance_m.dtype),
        tool_axis_progress=tool_axis_progress,
        final_success=(success & update_mask).to(distance_m.dtype),
        safety_failure=(safety_failure & update_mask).to(distance_m.dtype),
    )
