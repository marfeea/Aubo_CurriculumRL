"""阶段 E 的版本化课程配置；不复制或修改最终任务阈值。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

FINAL_TASK_THRESHOLDS_ARE_IMMUTABLE: Final = True
"""课程不得修改 ``configs.task`` 中的停靠或安全真值。"""

CURRICULUM_CONFIG_VERSION: Final = "stage-e-v1-provisional"
"""阶段 D 尚无足量 PPO 基线，门槛须在首次正式训练前复核。"""


@dataclass(frozen=True)
class AuxiliaryRewardScales:
    """仅缩放训练塑形项；最终成功和安全失败项始终保持原权重。"""

    distance_progress: float = 1.0
    best_progress: float = 1.0
    proximity: float = 1.0
    inner_docking_quality: float = 1.0
    low_speed_parking: float = 1.0
    first_entry: float = 1.0
    tool_axis_progress: float = 1.0


@dataclass(frozen=True)
class CurriculumLevelCfg:
    """一个等级可控制的训练分布和辅助奖励，不含任务真值。"""

    target_state_probabilities: tuple[float, float, float, float]
    auxiliary_rewards: AuxiliaryRewardScales
    approved_randomization: tuple[str, ...] = ()


@dataclass(frozen=True)
class CurriculumTransitionCfg:
    """滚动窗口、升降级和冷却规则。"""

    rolling_window_episodes: int = 64
    min_episodes_for_transition: int = 32
    promote_success_rate: float = 0.75
    max_promote_safety_failure_rate: float = 0.05
    demote_success_rate: float = 0.30
    max_demote_safety_failure_rate: float = 0.20
    cooldown_episodes: int = 32


@dataclass(frozen=True)
class CurriculumCfg:
    version: str
    levels: tuple[CurriculumLevelCfg, ...]
    transition: CurriculumTransitionCfg


CURRICULUM_CFG: Final = CurriculumCfg(
    version=CURRICULUM_CONFIG_VERSION,
    levels=(
        CurriculumLevelCfg(
            target_state_probabilities=(1.0, 0.0, 0.0, 0.0),
            auxiliary_rewards=AuxiliaryRewardScales(
                inner_docking_quality=0.25, low_speed_parking=0.25, tool_axis_progress=0.0
            ),
        ),
        CurriculumLevelCfg(
            target_state_probabilities=(1.0, 0.0, 0.0, 0.0),
            auxiliary_rewards=AuxiliaryRewardScales(low_speed_parking=0.5, tool_axis_progress=1.0),
        ),
        CurriculumLevelCfg(
            target_state_probabilities=(1.0, 0.0, 0.0, 0.0),
            auxiliary_rewards=AuxiliaryRewardScales(),
        ),
        CurriculumLevelCfg(
            target_state_probabilities=(0.25, 0.25, 0.25, 0.25),
            auxiliary_rewards=AuxiliaryRewardScales(),
        ),
        CurriculumLevelCfg(
            target_state_probabilities=(0.25, 0.25, 0.25, 0.25),
            auxiliary_rewards=AuxiliaryRewardScales(),
            # 尚无经资产/接触验证批准的物理随机化，故明确保持为空。
            approved_randomization=(),
        ),
    ),
    transition=CurriculumTransitionCfg(),
)
