"""阶段 D 奖励历史、首次事件和有限值纯逻辑测试。"""

import sys
from pathlib import Path

import torch

PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "source" / "CurriculumRL"
sys.path.insert(0, str(PACKAGE_SOURCE))

from CurriculumRL.logic.rewards import create_reward_state, reset_reward_state, update_reward_state  # noqa: E402


def update(state, distance, alignment, inside=False, success=False, unsafe=False):
    value = torch.tensor([distance], dtype=torch.float64)
    return update_reward_state(
        state,
        value,
        torch.tensor([0.1], dtype=torch.float64),
        torch.tensor([0.01], dtype=torch.float64),
        torch.tensor([alignment], dtype=torch.float64),
        torch.tensor([inside]),
        torch.tensor([success]),
        torch.tensor([unsafe]),
        proximity_length_scale_m=0.15,
        orientation_quality_scale_rad=0.35,
        speed_quality_scale_m_s=0.05,
    )


def test_first_sample_has_no_artificial_progress_and_all_components_are_finite() -> None:
    state = create_reward_state(1)
    components = update(state, 0.5, 0.2)
    assert components.distance_progress.item() == 0.0
    assert components.best_progress.item() == 0.0
    assert components.tool_axis_progress.item() == 0.0
    for value in vars(components).values():
        assert torch.isfinite(value).all()


def test_progress_and_first_entry_only_reward_new_events() -> None:
    state = create_reward_state(1)
    update(state, 0.5, 0.2)
    improved = update(state, 0.4, 0.4, inside=True)
    assert improved.distance_progress.item() > 0.0
    assert improved.best_progress.item() > 0.0
    assert improved.tool_axis_progress.item() > 0.0
    assert improved.first_entry.item() == 1.0
    repeated = update(state, 0.42, 0.3, inside=True)
    assert repeated.best_progress.item() == 0.0
    assert repeated.tool_axis_progress.item() == 0.0
    assert repeated.first_entry.item() == 0.0


def test_partial_reset_clears_only_selected_reward_history() -> None:
    state = create_reward_state(2)
    update_reward_state(
        state,
        torch.tensor([0.5, 0.6]),
        torch.tensor([0.1, 0.1]),
        torch.tensor([0.01, 0.01]),
        torch.tensor([0.2, 0.3]),
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        torch.tensor([False, False]),
        proximity_length_scale_m=0.15,
        orientation_quality_scale_rad=0.35,
        speed_quality_scale_m_s=0.05,
    )
    reset_reward_state(state, torch.tensor([1]))
    assert torch.isfinite(state.previous_distance_m[0])
    assert torch.isnan(state.previous_distance_m[1])
    assert state.has_entered.tolist() == [True, False]
