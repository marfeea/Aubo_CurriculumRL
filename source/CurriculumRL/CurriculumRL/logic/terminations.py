"""TCP 停靠安全失败条件的纯张量判定。"""

from __future__ import annotations

import torch


def outside_workspace(tcp_position_b: torch.Tensor, workspace_b: torch.Tensor) -> torch.Tensor:
    return torch.any((tcp_position_b < workspace_b[:, 0]) | (tcp_position_b > workspace_b[:, 1]), dim=-1)


def illegal_contact(
    contact_forces_w: torch.Tensor,
    body_names: tuple[str, ...],
    ignored_body_names: tuple[str, ...],
    force_threshold_n: float,
) -> torch.Tensor:
    """按 ``(env, body, xyz)`` 解析净接触力，忽略指定刚体。"""

    if contact_forces_w.ndim != 3 or contact_forces_w.shape[1] != len(body_names) or contact_forces_w.shape[2] != 3:
        raise ValueError(
            f"contact_forces_w 必须为 (env, body, 3)，实际 {tuple(contact_forces_w.shape)}，body={body_names}"
        )
    included = torch.tensor(
        [name not in ignored_body_names for name in body_names], dtype=torch.bool, device=contact_forces_w.device
    )
    force_magnitudes = torch.linalg.vector_norm(contact_forces_w[:, included], dim=-1)
    return torch.any(force_magnitudes > force_threshold_n, dim=-1)


def target_disturbed(
    target_position_w: torch.Tensor,
    target_baseline_position_w: torch.Tensor,
    target_linear_velocity_w: torch.Tensor,
    max_displacement_m: float,
    max_linear_speed_m_s: float,
) -> torch.Tensor:
    displacement = torch.linalg.vector_norm(target_position_w - target_baseline_position_w, dim=-1)
    speed = torch.linalg.vector_norm(target_linear_velocity_w, dim=-1)
    return (displacement > max_displacement_m) | (speed > max_linear_speed_m_s)


def timed_out(policy_step: torch.Tensor, max_policy_steps: int) -> torch.Tensor:
    return policy_step >= max_policy_steps
