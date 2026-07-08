"""不接入训练框架的课程统计与等级转换。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumState:
    level: int = 0
    episodes: int = 0
    successes: int = 0
    safety_failures: int = 0
    transition_reason: str = "initial"

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    @property
    def safety_failure_rate(self) -> float:
        return self.safety_failures / self.episodes if self.episodes else 0.0


def submit_episode(
    state: CurriculumState,
    *,
    success: bool,
    safety_failure: bool,
    min_episodes: int,
    promote_success_rate: float,
    max_promote_safety_failure_rate: float,
    demote_success_rate: float,
    max_level: int,
) -> CurriculumState:
    """累计当前窗口，并在门槛满足时晋级或回退后清零窗口。"""

    episodes = state.episodes + 1
    successes = state.successes + int(success)
    safety_failures = state.safety_failures + int(safety_failure)
    if episodes < min_episodes:
        return CurriculumState(state.level, episodes, successes, safety_failures, "accumulating")
    success_rate = successes / episodes
    safety_rate = safety_failures / episodes
    if success_rate >= promote_success_rate and safety_rate <= max_promote_safety_failure_rate:
        return CurriculumState(min(state.level + 1, max_level), transition_reason="promoted")
    if success_rate < demote_success_rate and state.level > 0:
        return CurriculumState(state.level - 1, transition_reason="demoted")
    return CurriculumState(state.level, transition_reason="window_retained_level")
