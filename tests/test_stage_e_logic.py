"""阶段 E 课程窗口、等级归属和配置恢复的纯逻辑测试。"""

import sys
from dataclasses import replace
from pathlib import Path

PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "source" / "CurriculumRL"
sys.path.insert(0, str(PACKAGE_SOURCE))

from CurriculumRL.configs.curriculum import CURRICULUM_CFG, CurriculumTransitionCfg  # noqa: E402
from CurriculumRL.logic.curriculum_state import CurriculumController, EpisodeResult  # noqa: E402


def episode(level: int, curriculum_success: bool, safety_failure: bool = False) -> EpisodeResult:
    return EpisodeResult(
        level=level,
        curriculum_success=curriculum_success,
        formal_parking_success=curriculum_success,
        safety_failure=safety_failure,
        target_state_index=0,
        path_mode_index=0,
        collision_profile_id="C1",
    )

def test_safety_failure_overrides_success_and_promotes_only_after_complete_batch() -> None:
    config = replace(
        CURRICULUM_CFG,
        transition=CurriculumTransitionCfg(
            rolling_window_episodes=4,
            min_episodes_for_transition=4,
            promote_success_rate=0.75,
            max_promote_safety_failure_rate=0.0,
            demote_success_rate=0.25,
            max_demote_safety_failure_rate=0.2,
            cooldown_episodes=0,
        ),
    )
    controller = CurriculumController(config)
    controller.submit_batch(
        (
            episode(0, True),
            episode(0, True),
            episode(0, True, safety_failure=True),
        )
    )
    assert controller.level == 0
    transition = controller.submit_batch((episode(0, True),))
    assert transition is None
    assert controller.windows[0][2].success is False
    assert controller.windows[0][2].safety_failure is True


def test_old_level_episode_does_not_pollute_new_level_window_after_promotion() -> None:
    config = replace(
        CURRICULUM_CFG,
        transition=CurriculumTransitionCfg(
            rolling_window_episodes=2,
            min_episodes_for_transition=2,
            promote_success_rate=1.0,
            max_promote_safety_failure_rate=0.0,
            demote_success_rate=0.25,
            max_demote_safety_failure_rate=0.2,
            cooldown_episodes=0,
        ),
    )
    controller = CurriculumController(config)
    transition = controller.submit_batch(
        (
            episode(0, True),
            episode(0, True),
        )
    )
    assert transition is not None and transition.new_level == 1
    controller.submit_batch((episode(0, False, safety_failure=True),))
    assert controller.level == 1
    assert len(controller.windows[0]) == 2
    assert len(controller.windows.get(1, ())) == 0


def test_snapshot_restores_fixed_windows_and_rejects_wrong_version() -> None:
    controller = CurriculumController(CURRICULUM_CFG)
    controller.submit_batch((episode(0, True),))
    snapshot = controller.snapshot()
    restored = CurriculumController.from_snapshot(CURRICULUM_CFG, snapshot)
    assert restored.snapshot() == snapshot
    snapshot["config_version"] = "wrong-version"
    try:
        CurriculumController.from_snapshot(CURRICULUM_CFG, snapshot)
    except ValueError as error:
        assert "版本" in str(error)
    else:
        raise AssertionError("配置版本不一致必须快速失败")


def test_every_level_has_normalized_four_state_distribution() -> None:
    for level in CURRICULUM_CFG.levels:
        assert len(level.target_state_probabilities) == 4
        assert sum(level.target_state_probabilities) == 1.0
