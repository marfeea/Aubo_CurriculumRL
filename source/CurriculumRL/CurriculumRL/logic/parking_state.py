"""带迟滞、低速、姿态和 dwell 的统一停车状态机。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParkingState:
    inside: torch.Tensor
    dwell_steps: torch.Tensor
    last_control_step: torch.Tensor
    success: torch.Tensor


def initial_parking_state(num_envs: int, *, device: torch.device | str = "cpu") -> ParkingState:
    return ParkingState(
        inside=torch.zeros(num_envs, dtype=torch.bool, device=device),
        dwell_steps=torch.zeros(num_envs, dtype=torch.long, device=device),
        last_control_step=torch.full((num_envs,), -1, dtype=torch.long, device=device),
        success=torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def update_parking_state(
    state: ParkingState,
    distance_m: torch.Tensor,
    tcp_speed_m_s: torch.Tensor,
    orientation_error_rad: torch.Tensor,
    control_step: torch.Tensor,
    *,
    enter_distance_m: float,
    exit_distance_m: float,
    max_tcp_speed_m_s: float,
    max_orientation_error_rad: float,
    required_dwell_steps: int,
) -> ParkingState:
    """提交一次控制步；重复提交相同 step 时保持状态不变。"""

    if enter_distance_m >= exit_distance_m:
        raise ValueError("进入距离必须小于退出距离")
    if required_dwell_steps < 1:
        raise ValueError("required_dwell_steps 必须大于零")
    new_step = control_step != state.last_control_step
    next_inside = torch.where(state.inside, distance_m <= exit_distance_m, distance_m <= enter_distance_m)
    next_inside = torch.where(new_step, next_inside, state.inside)
    qualifies = (
        next_inside & (tcp_speed_m_s <= max_tcp_speed_m_s) & (orientation_error_rad <= max_orientation_error_rad)
    )
    candidate_dwell = torch.where(qualifies, state.dwell_steps + 1, torch.zeros_like(state.dwell_steps))
    next_dwell = torch.where(new_step, candidate_dwell, state.dwell_steps)
    next_success = next_dwell >= required_dwell_steps
    return ParkingState(
        inside=next_inside,
        dwell_steps=next_dwell,
        last_control_step=torch.where(new_step, control_step, state.last_control_step),
        success=torch.where(new_step, next_success, state.success),
    )
