"""阶段 E 的版本化课程配置；不复制或修改最终任务阈值。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

FINAL_TASK_THRESHOLDS_ARE_IMMUTABLE: Final = True
"""课程不得修改 ``configs.task`` 中的停靠或安全真值。"""

CURRICULUM_CONFIG_VERSION: Final = "stage-e-v1-provisional"
"""阶段 D 尚无足量 PPO 基线，门槛须在首次正式训练前复核。"""

CURRICULUM_V2_SCHEMA_VERSION: Final = "curriculum-v2-p0"
"""P0 冻结的下一代策略接口版本；尚未接入当前阶段 E 训练行为。"""


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


@dataclass(frozen=True)
class ObservationFieldCfg:
    """一项固定策略观测的张量契约。"""

    name: str
    dimension: int
    unit: str
    coordinate_frame: str
    normalization: str
    enabled_levels: tuple[int, ...]


@dataclass(frozen=True)
class CurriculumOutcomeSchemaCfg:
    """课程统计与正式任务真值的结果字段及其安全优先级。"""

    curriculum_success_field: str
    formal_parking_success_field: str
    safety_failure_field: str
    precedence: tuple[str, ...]


@dataclass(frozen=True)
class PathModeCfg:
    """P0 冻结的路径条件变量；P3 再将其接入轨迹约束。"""

    name: str
    one_hot_index: int
    sampling_probability: float
    midpoint_offset_b: tuple[float, float, float]


CURRICULUM_V2_PATH_MODES: Final = (
    PathModeCfg("direct", 0, 1.0 / 3.0, (0.0, 0.0, 0.0)),
    PathModeCfg("lateral_positive", 1, 1.0 / 3.0, (0.0, 0.12, 0.0)),
    PathModeCfg("lateral_negative", 2, 1.0 / 3.0, (0.0, -0.12, 0.0)),
)


CURRICULUM_V2_OBSERVATION_SCHEMA: Final = (
    # 当前 29 维字段保持原有语义，避免把阶段 E checkpoint 误接入新谱系。
    ObservationFieldCfg("arm_joint_position", 6, "rad", "joint", "joint_limit_relative", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("arm_joint_velocity", 6, "rad/s", "joint", "velocity_limit_relative", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("tcp_position_error", 3, "m", "B", "workspace_relative", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("tcp_orientation_error", 3, "rad", "B", "pi_relative", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("tcp_linear_velocity", 3, "m/s", "B", "velocity_limit_relative", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("tcp_angular_velocity", 3, "rad/s", "B", "velocity_limit_relative", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("target_state_one_hot", 4, "one_hot", "none", "identity", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("curriculum_level_normalized", 1, "ratio", "none", "[0,1]", (1, 2, 3, 4, 5)),
    # P0 新增的预留字段从阶段 1 起始终输出；未启用信息严格置零并由掩码标识。
    ObservationFieldCfg("path_mode_one_hot", 3, "one_hot", "B", "identity", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("collision_profile_one_hot", 3, "one_hot", "none", "identity", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("collision_group_min_clearance", 3, "m", "E", "clearance_cap_relative", (4, 5)),
    ObservationFieldCfg("collision_clearance_enabled_mask", 3, "binary", "none", "identity", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("tcp_action_mask", 6, "binary", "B", "identity", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("curriculum_stage_one_hot", 5, "one_hot", "none", "identity", (1, 2, 3, 4, 5)),
)

CURRICULUM_V2_OBSERVATION_DIMENSION: Final = sum(field.dimension for field in CURRICULUM_V2_OBSERVATION_SCHEMA)

CURRICULUM_V2_OUTCOME_SCHEMA: Final = CurriculumOutcomeSchemaCfg(
    curriculum_success_field="curriculum_success",
    formal_parking_success_field="formal_parking_success",
    safety_failure_field="safety_failure",
    precedence=("safety_failure", "formal_parking_success", "curriculum_success"),
)


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
