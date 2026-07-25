"""不依赖 Isaac 的阶段成功状态机。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..configs.curriculum import StageSuccessCfg


@dataclass
class CurriculumSuccessState:
    """逐环境阶段成功缓存，与正式停车状态机彼此独立。"""

    dwell_steps: torch.Tensor
    last_control_step: torch.Tensor
    success: torch.Tensor


def initial_curriculum_success_state(
    num_envs: int, *, device: torch.device | str | None = None
) -> CurriculumSuccessState:
    return CurriculumSuccessState(
        dwell_steps=torch.zeros(num_envs, dtype=torch.long, device=device),
        last_control_step=torch.full((num_envs,), -1, dtype=torch.long, device=device),
        success=torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def update_curriculum_success(
    state: CurriculumSuccessState,
    position_error_m: torch.Tensor,
    orientation_error_rad: torch.Tensor,
    tcp_speed_m_s: torch.Tensor,
    safety_failure: torch.Tensor,
    formal_parking_success: torch.Tensor,
    control_step: torch.Tensor,
    config: StageSuccessCfg,
    path_reference_reached: torch.Tensor | None = None,
) -> CurriculumSuccessState:
    """按显式阶段门槛更新 dwell；安全失败始终覆盖两种成功。"""

    values = (
        position_error_m,
        orientation_error_rad,
        tcp_speed_m_s,
        safety_failure,
        formal_parking_success,
        control_step,
    )
    if any(value.ndim != 1 for value in values):
        raise ValueError("阶段成功输入必须是逐环境一维张量")
    if any(value.shape != position_error_m.shape for value in values):
        raise ValueError("阶段成功输入形状必须一致")
    if config.required_dwell_steps <= 0:
        raise ValueError("阶段成功 dwell 步数必须为正")
    fresh = control_step != state.last_control_step
    qualified = (position_error_m <= config.max_position_error_m) & (tcp_speed_m_s <= config.max_tcp_speed_m_s)
    if config.max_orientation_error_rad is not None:
        qualified &= orientation_error_rad <= config.max_orientation_error_rad
    if config.require_formal_parking_success:
        qualified &= formal_parking_success
    if config.require_path_reference_reached:
        if path_reference_reached is None or path_reference_reached.shape != position_error_m.shape:
            raise ValueError("路径阶段成功必须提供同形状的参考点到达状态")
        qualified &= path_reference_reached
    qualified &= ~safety_failure
    next_dwell = torch.where(qualified, state.dwell_steps + 1, torch.zeros_like(state.dwell_steps))
    state.dwell_steps = torch.where(fresh, next_dwell, state.dwell_steps)
    state.success = torch.where(
        fresh,
        (state.dwell_steps >= config.required_dwell_steps) & ~safety_failure,
        state.success & ~safety_failure,
    )
    state.last_control_step = torch.where(fresh, control_step, state.last_control_step)
    return state
