from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "CurriculumRL"))

from CurriculumRL.logic.stage_p4_evaluation import (  # noqa: E402
    Stage1EvaluationUnit,
    cluster_summary,
    resample_tcp_trajectory,
    stage1_evaluation_units,
    summarize_episode_behavior,
    trajectory_cluster_labels,
    unit_result_filename,
)


def test_stage1_evaluation_units_are_seed_deduplicated_and_mode_complete() -> None:
    units = stage1_evaluation_units("ppo", (7, 11, 7), 3)
    assert units == (
        Stage1EvaluationUnit("ppo", 7, 0),
        Stage1EvaluationUnit("ppo", 11, 0),
        Stage1EvaluationUnit("ppo", 7, 1),
        Stage1EvaluationUnit("ppo", 11, 1),
        Stage1EvaluationUnit("ppo", 7, 2),
        Stage1EvaluationUnit("ppo", 11, 2),
    )
    assert unit_result_filename(units[-1]) == "ppo_seed_11_path_2.json"


def test_trajectory_clusters_use_normalized_time_not_raw_episode_length() -> None:
    direct_short = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    direct_long = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    lateral = ((0.0, 0.0, 0.0), (0.5, 0.2, 0.0), (1.0, 0.0, 0.0))
    assert resample_tcp_trajectory(direct_short, 5).shape == (5, 3)
    labels = trajectory_cluster_labels(
        (direct_short, direct_long, lateral), point_count=9, mean_distance_threshold_m=0.03
    )
    assert labels == (0, 0, 1)
    assert cluster_summary(labels) == {"0": 2, "1": 1}


def test_invalid_p4_evaluation_inputs_fail_fast() -> None:
    with pytest.raises(ValueError, match="策略条件"):
        stage1_evaluation_units("unknown", (7,), 3)
    with pytest.raises(ValueError, match="至少两个"):
        resample_tcp_trajectory(((0.0, 0.0, 0.0),), 4)
    with pytest.raises(ValueError, match="阈值"):
        trajectory_cluster_labels((((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), point_count=4, mean_distance_threshold_m=0.0)


def test_episode_behavior_summary_exposes_action_progress_and_stage_dwell() -> None:
    summary = summarize_episode_behavior(
        raw_actions=(
            (1.0, 0.0, 0.0, 0.5, 0.0, 0.0),
            (0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
            (0.25, 0.0, 0.0, 0.5, 0.0, 0.0),
        ),
        processed_actions=(
            (0.02, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.01, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.005, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        tcp_positions_b=((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)),
        position_error_vectors_b=((0.2, 0.0, 0.0), (0.07, 0.0, 0.0), (0.05, 0.0, 0.0)),
        position_errors_m=(0.2, 0.07, 0.05),
        tcp_speeds_m_s=(0.1, 0.05, 0.04),
        path_reference_distances_m=(0.2, 0.03, 0.08),
        path_reference_reached=(False, True, True),
        controller_protections={
            "singular": (False, True, False),
            "delta_limited": (True, False, False),
            "velocity_limited": (False, False, False),
            "position_limited": (False, False, False),
        },
        terminal_position_error_m=0.05,
        terminal_tcp_speed_m_s=0.04,
        max_position_error_m=0.08,
        max_tcp_speed_m_s=0.08,
    )
    assert summary["minimum_position_error_m"] == pytest.approx(0.05)
    assert summary["minimum_position_error_step"] == 3
    assert summary["first_path_reference_reached_step"] == 2
    assert summary["stage_qualified_step_count"] == 2
    assert summary["stage_qualified_longest_run"] == 2
    assert summary["tcp_path_length_m"] == pytest.approx(0.2)
    assert summary["translation_action_target_alignment_mean"] == pytest.approx(1.0)
    assert summary["controller_protection_rate_per_policy_step"]["singular"] == pytest.approx(1.0 / 3.0)
    assert summary["processed_action"]["max_abs"][3:] == [0.0, 0.0, 0.0]
