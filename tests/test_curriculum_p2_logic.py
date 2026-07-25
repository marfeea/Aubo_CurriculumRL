"""P2 五阶段公共接口、阶段成功与快照拒绝测试。"""

import sys
from pathlib import Path

import pytest
import torch

PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "source" / "CurriculumRL"
sys.path.insert(0, str(PACKAGE_SOURCE))

from CurriculumRL.configs.curriculum import (  # noqa: E402
    CURRICULUM_CFG,
    CURRICULUM_COLLISION_PROFILES,
    CURRICULUM_V2_OBSERVATION_DIMENSION,
)
from CurriculumRL.logic.curriculum_state import CurriculumController, EpisodeResult  # noqa: E402
from CurriculumRL.logic.curriculum_success import (  # noqa: E402
    initial_curriculum_success_state,
    update_curriculum_success,
)


def _update(
    state,
    *,
    distance: float,
    orientation: float,
    speed: float,
    safety: bool,
    formal: bool,
    step: int,
    path_reference_reached: bool = True,
):
    return update_curriculum_success(
        state,
        torch.tensor([distance]),
        torch.tensor([orientation]),
        torch.tensor([speed]),
        torch.tensor([safety]),
        torch.tensor([formal]),
        torch.tensor([step]),
        CURRICULUM_CFG.levels[0].stage_success,
        torch.tensor([path_reference_reached]),
    )


def test_stage_success_uses_its_own_thresholds_and_safety_overrides() -> None:
    state = initial_curriculum_success_state(1)
    state = _update(state, distance=0.07, orientation=3.14, speed=0.07, safety=False, formal=False, step=0)
    assert state.success.tolist() == [False]
    state = _update(state, distance=0.07, orientation=3.14, speed=0.07, safety=False, formal=False, step=1)
    assert state.success.tolist() == [True]
    state = _update(state, distance=0.01, orientation=0.0, speed=0.0, safety=True, formal=True, step=2)
    assert state.success.tolist() == [False]


def test_duplicate_control_step_does_not_increment_stage_dwell() -> None:
    state = initial_curriculum_success_state(1)
    state = _update(state, distance=0.07, orientation=0.0, speed=0.07, safety=False, formal=False, step=0)
    state = _update(state, distance=0.07, orientation=0.0, speed=0.07, safety=False, formal=False, step=0)
    assert state.dwell_steps.tolist() == [1]


def test_stage_one_success_requires_its_path_reference() -> None:
    state = initial_curriculum_success_state(1)
    state = _update(
        state, distance=0.07, orientation=0.0, speed=0.07, safety=False, formal=False, step=0, path_reference_reached=False
    )
    assert state.success.tolist() == [False]


def test_five_levels_have_fixed_observation_contract_and_episode_metadata() -> None:
    assert CURRICULUM_V2_OBSERVATION_DIMENSION == 52
    assert tuple(profile.identifier for profile in CURRICULUM_COLLISION_PROFILES) == ("C1", "C2", "C3")
    assert tuple(level.collision_profile_id for level in CURRICULUM_CFG.levels) == ("C1", "C1", "C2", "C3", "C3")
    assert all(level.collision_clearance_enabled_mask == (0.0, 0.0, 0.0) for level in CURRICULUM_CFG.levels)
    assert CURRICULUM_CFG.levels[0].tcp_action_mask == (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert all(sum(level.path_mode_probabilities) == 1.0 for level in CURRICULUM_CFG.levels)


def test_snapshot_rejects_missing_p2_schema_and_records_both_successes() -> None:
    controller = CurriculumController(CURRICULUM_CFG)
    controller.submit_batch(
        (
            EpisodeResult(0, True, False, False, 0, 1, "C1"),
            EpisodeResult(0, True, True, False, 0, 2, "C1"),
        )
    )
    snapshot = controller.snapshot()
    assert snapshot["observation_schema_version"] == CURRICULUM_CFG.observation_schema_version
    assert snapshot["windows"]["0"][1]["formal_parking_success"] is True
    snapshot.pop("observation_schema_version")
    with pytest.raises(ValueError, match="schema"):
        CurriculumController.from_snapshot(CURRICULUM_CFG, snapshot)
