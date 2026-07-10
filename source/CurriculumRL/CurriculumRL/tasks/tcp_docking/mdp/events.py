"""阶段 E 课程目标采样 reset 事件。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .runtime_state import reset_curriculum_runtime

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_curriculum_target(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """按旧 episode 的归属等级结算后，为下一回合采样目标。"""

    reset_curriculum_runtime(env, env_ids)
