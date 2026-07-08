"""阶段 B 目标 reset 的 Isaac 写入与回读适配。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...configs.assets import SCENE_ENTITY_TARGET
from ...configs.task import (
    FLANGE_TO_TOOL_ROTATION_F,
    TARGET_STATES,
    TARGET_TO_TOOL_ROTATION_T,
)
from ...logic.reset_state import ResetCache, commit_target_readback, prepare_target_reset

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene


def reset_targets(
    scene: InteractiveScene,
    cache: ResetCache,
    env_ids: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    """对指定环境写入目标位姿/零速度，并将引擎缓冲回读为回合基准。"""

    target = scene[SCENE_ENTITY_TARGET]
    device = scene.env_origins.device
    dtype = scene.env_origins.dtype
    target_pose_w, target_velocity_w = prepare_target_reset(
        cache,
        env_ids,
        state_indices,
        scene.env_origins,
        torch.tensor([state.position_e for state in TARGET_STATES], dtype=dtype, device=device),
        torch.tensor([state.rotation_wxyz for state in TARGET_STATES], dtype=dtype, device=device),
        torch.tensor([state.preposition_e for state in TARGET_STATES], dtype=dtype, device=device),
        torch.tensor(TARGET_TO_TOOL_ROTATION_T, dtype=dtype, device=device),
        torch.tensor(FLANGE_TO_TOOL_ROTATION_F, dtype=dtype, device=device),
    )
    target.write_root_pose_to_sim(target_pose_w, env_ids=env_ids)
    target.write_root_velocity_to_sim(target_velocity_w, env_ids=env_ids)
    commit_target_readback(
        cache,
        env_ids,
        target.data.root_pos_w[env_ids].clone(),
        target.data.root_quat_w[env_ids].clone(),
        torch.tensor(TARGET_TO_TOOL_ROTATION_T, dtype=dtype, device=device),
        torch.tensor(FLANGE_TO_TOOL_ROTATION_F, dtype=dtype, device=device),
    )
