"""目标 reset 选择、坐标变换和逐环境缓存的纯逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from .orientation import desired_flange_orientation


@dataclass
class ResetCache:
    target_state_index: torch.Tensor
    target_position_w: torch.Tensor
    target_quaternion_wxyz: torch.Tensor
    target_baseline_position_w: torch.Tensor
    target_baseline_quaternion_wxyz: torch.Tensor
    preposition_w: torch.Tensor
    desired_flange_quaternion_wxyz: torch.Tensor
    best_distance_m: torch.Tensor
    milestone_reached: torch.Tensor
    orientation_progress: torch.Tensor
    parking_inside: torch.Tensor
    dwell_steps: torch.Tensor
    last_control_step: torch.Tensor


def create_reset_cache(num_envs: int, *, device: torch.device | str = "cpu") -> ResetCache:
    zeros = torch.zeros(num_envs, device=device)
    return ResetCache(
        target_state_index=torch.zeros(num_envs, dtype=torch.long, device=device),
        target_position_w=torch.zeros((num_envs, 3), device=device),
        target_quaternion_wxyz=torch.zeros((num_envs, 4), device=device),
        target_baseline_position_w=torch.zeros((num_envs, 3), device=device),
        target_baseline_quaternion_wxyz=torch.zeros((num_envs, 4), device=device),
        preposition_w=torch.zeros((num_envs, 3), device=device),
        desired_flange_quaternion_wxyz=torch.zeros((num_envs, 4), device=device),
        best_distance_m=torch.full((num_envs,), torch.inf, device=device),
        milestone_reached=torch.zeros(num_envs, dtype=torch.bool, device=device),
        orientation_progress=zeros.clone(),
        parking_inside=torch.zeros(num_envs, dtype=torch.bool, device=device),
        dwell_steps=torch.zeros(num_envs, dtype=torch.long, device=device),
        last_control_step=torch.full((num_envs,), -1, dtype=torch.long, device=device),
    )


def prepare_target_reset(
    cache: ResetCache,
    env_ids: torch.Tensor,
    state_indices: torch.Tensor,
    env_origins_w: torch.Tensor,
    target_positions_e: torch.Tensor,
    target_quaternions_wxyz: torch.Tensor,
    prepositions_e: torch.Tensor,
    target_to_tool_rotation_t: torch.Tensor,
    flange_to_tool_rotation_f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """只更新 ``env_ids`` 缓存，返回应写入仿真的目标位姿和零速度。"""

    if env_ids.ndim != 1 or state_indices.shape != env_ids.shape:
        raise ValueError("env_ids 与 state_indices 必须是一维同形张量")
    positions_w = target_positions_e[state_indices] + env_origins_w[env_ids]
    quaternions = target_quaternions_wxyz[state_indices]
    prepositions_w = prepositions_e[state_indices] + env_origins_w[env_ids]
    cache.target_state_index[env_ids] = state_indices
    cache.target_position_w[env_ids] = positions_w
    cache.target_quaternion_wxyz[env_ids] = quaternions
    cache.preposition_w[env_ids] = prepositions_w
    cache.desired_flange_quaternion_wxyz[env_ids] = desired_flange_orientation(
        quaternions, target_to_tool_rotation_t, flange_to_tool_rotation_f
    )
    cache.best_distance_m[env_ids] = torch.inf
    cache.milestone_reached[env_ids] = False
    cache.orientation_progress[env_ids] = 0.0
    cache.parking_inside[env_ids] = False
    cache.dwell_steps[env_ids] = 0
    cache.last_control_step[env_ids] = -1
    target_pose_w = torch.cat((positions_w, quaternions), dim=-1)
    zero_velocity_w = torch.zeros((env_ids.numel(), 6), dtype=positions_w.dtype, device=positions_w.device)
    return target_pose_w, zero_velocity_w


def commit_target_readback(
    cache: ResetCache,
    env_ids: torch.Tensor,
    actual_position_w: torch.Tensor,
    actual_quaternion_wxyz: torch.Tensor,
    target_to_tool_rotation_t: torch.Tensor,
    flange_to_tool_rotation_f: torch.Tensor,
) -> None:
    """将物理引擎回读位姿作为本回合目标基准。"""

    cache.target_position_w[env_ids] = actual_position_w
    cache.target_quaternion_wxyz[env_ids] = actual_quaternion_wxyz
    cache.target_baseline_position_w[env_ids] = actual_position_w
    cache.target_baseline_quaternion_wxyz[env_ids] = actual_quaternion_wxyz
    cache.desired_flange_quaternion_wxyz[env_ids] = desired_flange_orientation(
        actual_quaternion_wxyz, target_to_tool_rotation_t, flange_to_tool_rotation_f
    )


def clone_reset_cache(cache: ResetCache) -> ResetCache:
    """测试和事务检查使用的深张量副本。"""

    return ResetCache(**{field.name: getattr(cache, field.name).clone() for field in fields(cache)})
