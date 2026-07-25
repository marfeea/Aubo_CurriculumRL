"""P5 位姿阶段配置与固定评估矩阵的纯逻辑测试。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "CurriculumRL"))

from CurriculumRL.configs.curriculum import CURRICULUM_CFG  # noqa: E402
from CurriculumRL.logic.rewards import create_reward_state, update_reward_state  # noqa: E402
from CurriculumRL.logic.stage_p5_evaluation import (  # noqa: E402
    Stage2EvaluationUnit,
    capability_name,
    stage2_evaluation_units,
    unit_result_filename,
)


def test_stage2_keeps_stage1_contract_and_releases_rotation_with_regression_sampling() -> None:
    stage1 = CURRICULUM_CFG.levels[0]
    stage2 = CURRICULUM_CFG.levels[1]
    assert stage2.target_state_probabilities == stage1.target_state_probabilities
    assert stage2.path_mode_probabilities == stage1.path_mode_probabilities
    assert stage2.collision_profile_id == stage1.collision_profile_id == "C1"
    assert stage1.tcp_action_mask == (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert stage2.tcp_action_mask == (1.0,) * 6
    assert stage2.previous_level_sampling_probability == 0.25
    assert stage2.stage_success.max_orientation_error_rad == pytest.approx(math.radians(20.0))


def test_stage2_pose_shaping_rewards_axis_progress_and_pose_holding_quality() -> None:
    state = create_reward_state(1)
    common = {
        "distance_m": torch.tensor([0.03]),
        "orientation_error_rad": torch.tensor([0.10]),
        "tcp_speed_m_s": torch.tensor([0.01]),
        "parking_inside": torch.tensor([True]),
        "success": torch.tensor([False]),
        "safety_failure": torch.tensor([False]),
        "proximity_length_scale_m": 0.15,
        "orientation_quality_scale_rad": 0.35,
        "speed_quality_scale_m_s": 0.05,
    }
    update_reward_state(state, axis_alignment=torch.tensor([0.5]), **common)
    rewards = update_reward_state(state, axis_alignment=torch.tensor([0.8]), **common)
    assert rewards.tool_axis_progress.item() == pytest.approx(0.3)
    assert 0.0 < rewards.inner_docking_quality.item() <= 1.0
    assert 0.0 < rewards.low_speed_parking.item() <= 1.0


def test_p5_matrix_covers_pose_and_fixed_stage1_regression() -> None:
    units = stage2_evaluation_units("ppo", (7, 11, 7), 3)
    assert len(units) == 12
    assert units[0] == Stage2EvaluationUnit("ppo", 7, 0, 1)
    assert units[-1] == Stage2EvaluationUnit("ppo", 11, 2, 0)
    assert capability_name(1) == "stage2_pose"
    assert capability_name(0) == "stage1_regression"
    assert unit_result_filename(units[-1]) == "ppo_level_0_seed_11_path_2.json"


def test_invalid_p5_matrix_inputs_fail_fast() -> None:
    with pytest.raises(ValueError, match="策略条件"):
        stage2_evaluation_units("unknown", (7,), 3)
    with pytest.raises(ValueError, match="L1 和 L0"):
        stage2_evaluation_units("zero", (7,), 3, (2,))
