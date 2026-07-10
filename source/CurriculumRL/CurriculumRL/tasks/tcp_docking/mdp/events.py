"""阶段 D 固定 L0 reset 事件。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .runtime_state import reset_runtime

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_fixed_l0_target(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """固定使用最简单的 state_01；阶段 E 才扩大目标分布。"""

    reset_runtime(env, env_ids, torch.zeros_like(env_ids))
