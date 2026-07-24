"""P0 冻结的下一代课程接口纯逻辑测试。"""

import sys
from pathlib import Path

PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "source" / "CurriculumRL"
sys.path.insert(0, str(PACKAGE_SOURCE))

from CurriculumRL.configs.curriculum import (  # noqa: E402
    CURRICULUM_V2_OBSERVATION_DIMENSION,
    CURRICULUM_V2_OBSERVATION_SCHEMA,
    CURRICULUM_V2_OUTCOME_SCHEMA,
    CURRICULUM_V2_PATH_MODES,
    CURRICULUM_V2_SCHEMA_VERSION,
)


def test_curriculum_v2_schema_is_versioned_and_has_fixed_shape() -> None:
    assert CURRICULUM_V2_SCHEMA_VERSION == "curriculum-v2-p0"
    assert CURRICULUM_V2_OBSERVATION_DIMENSION == 52
    assert len({field.name for field in CURRICULUM_V2_OBSERVATION_SCHEMA}) == len(CURRICULUM_V2_OBSERVATION_SCHEMA)
    assert all(field.dimension > 0 for field in CURRICULUM_V2_OBSERVATION_SCHEMA)
    assert all(field.enabled_levels == (1, 2, 3, 4, 5) or field.name == "collision_group_min_clearance"
               for field in CURRICULUM_V2_OBSERVATION_SCHEMA)


def test_curriculum_v2_uses_masks_for_disabled_information() -> None:
    fields = {field.name: field for field in CURRICULUM_V2_OBSERVATION_SCHEMA}
    assert fields["collision_group_min_clearance"].enabled_levels == (4, 5)
    assert fields["collision_clearance_enabled_mask"].enabled_levels == (1, 2, 3, 4, 5)
    assert fields["tcp_action_mask"].dimension == 6


def test_safety_failure_has_priority_over_both_success_signals() -> None:
    assert CURRICULUM_V2_OUTCOME_SCHEMA.precedence == (
        "safety_failure",
        "formal_parking_success",
        "curriculum_success",
    )


def test_path_modes_are_versioned_configuration_not_reward_side_effects() -> None:
    assert tuple(mode.name for mode in CURRICULUM_V2_PATH_MODES) == (
        "direct",
        "lateral_positive",
        "lateral_negative",
    )
    assert tuple(mode.one_hot_index for mode in CURRICULUM_V2_PATH_MODES) == (0, 1, 2)
    assert sum(mode.sampling_probability for mode in CURRICULUM_V2_PATH_MODES) == 1.0
    assert CURRICULUM_V2_PATH_MODES[1].midpoint_offset_b == (0.0, 0.12, 0.0)
    assert CURRICULUM_V2_PATH_MODES[2].midpoint_offset_b == (0.0, -0.12, 0.0)
