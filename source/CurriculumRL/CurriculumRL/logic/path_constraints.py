"""阶段 1 路径模式的纯几何约束与逐环境进度状态。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PathConstraintState:
    """中段参考点的最优距离和一次性到达标记。"""

    reference_point_b: torch.Tensor
    best_reference_distance_m: torch.Tensor
    reference_reached: torch.Tensor


@dataclass(frozen=True)
class PathConstraintMetrics:
    reference_distance_m: torch.Tensor
    reference_progress_m: torch.Tensor
    reference_reached_now: torch.Tensor
    reference_reached: torch.Tensor


def initial_path_constraint_state(num_envs: int, *, device: torch.device | str | None = None) -> PathConstraintState:
    return PathConstraintState(
        reference_point_b=torch.zeros((num_envs, 3), device=device),
        best_reference_distance_m=torch.full((num_envs,), torch.inf, device=device),
        reference_reached=torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def reset_path_constraint_state(
    state: PathConstraintState,
    env_ids: torch.Tensor,
    start_tcp_position_b: torch.Tensor,
    target_tcp_position_b: torch.Tensor,
    path_mode_indices: torch.Tensor,
    midpoint_offsets_b: torch.Tensor,
) -> None:
    """按 ``z`` 写入中段参考点；direct 的偏移严格为零。"""

    if start_tcp_position_b.shape != target_tcp_position_b.shape or start_tcp_position_b.shape[-1] != 3:
        raise ValueError("路径约束起点与终点必须为 (env, 3)")
    if path_mode_indices.shape != env_ids.shape:
        raise ValueError("路径模式索引必须与 env_ids 形状一致")
    if midpoint_offsets_b.shape != (3, 3):
        raise ValueError("路径模式偏移必须为 (3, 3)")
    if torch.any((path_mode_indices < 0) | (path_mode_indices >= midpoint_offsets_b.shape[0])):
        raise ValueError("路径模式索引越界")
    state.reference_point_b[env_ids] = (
        0.5 * (start_tcp_position_b[env_ids] + target_tcp_position_b[env_ids])
        + midpoint_offsets_b[path_mode_indices]
    )
    state.best_reference_distance_m[env_ids] = torch.inf
    state.reference_reached[env_ids] = False


def update_path_constraint(
    state: PathConstraintState,
    tcp_position_b: torch.Tensor,
    *,
    reference_reach_tolerance_m: float,
    update_mask: torch.Tensor | None = None,
) -> PathConstraintMetrics:
    """更新参考点进展；到达后保持，避免离开参考点撤销路径资格。"""

    if reference_reach_tolerance_m <= 0.0:
        raise ValueError("路径参考点到达容差必须为正")
    if tcp_position_b.shape != state.reference_point_b.shape:
        raise ValueError("TCP 位置与路径约束状态形状不一致")
    if update_mask is None:
        update_mask = torch.ones(tcp_position_b.shape[0], dtype=torch.bool, device=tcp_position_b.device)
    if update_mask.shape != state.reference_reached.shape:
        raise ValueError("路径约束更新掩码形状不一致")
    distance = torch.linalg.vector_norm(tcp_position_b - state.reference_point_b, dim=-1)
    best_valid = torch.isfinite(state.best_reference_distance_m) & update_mask
    progress = torch.where(best_valid, (state.best_reference_distance_m - distance).clamp_min(0.0), 0.0)
    reached_now = (distance <= reference_reach_tolerance_m) & ~state.reference_reached & update_mask
    state.best_reference_distance_m[update_mask] = torch.minimum(
        state.best_reference_distance_m[update_mask], distance[update_mask]
    )
    state.reference_reached |= reached_now
    return PathConstraintMetrics(distance, progress, reached_now, state.reference_reached.clone())
