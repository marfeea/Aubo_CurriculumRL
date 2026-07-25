"""不依赖 Isaac 的阶段 E 课程统计、转换与可恢复状态。"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..configs.curriculum import CURRICULUM_COLLISION_PROFILES, CurriculumCfg


@dataclass(frozen=True)
class CurriculumState:
    """保留阶段 B 的累计接口，供已有纯逻辑调用继续使用。"""

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
    """阶段 B 兼容的累计窗口转换函数。"""

    episodes = state.episodes + 1
    successes = state.successes + int(success and not safety_failure)
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


@dataclass(frozen=True)
class EpisodeResult:
    """一个完整 episode 的归属等级和最终结果。"""

    level: int
    curriculum_success: bool
    formal_parking_success: bool
    safety_failure: bool
    target_state_index: int
    path_mode_index: int
    collision_profile_id: str

    @property
    def success(self) -> bool:
        """课程窗口使用的成功信号；保留只读别名便于统计调用。"""

        return self.curriculum_success and not self.safety_failure


@dataclass(frozen=True)
class CurriculumTransition:
    old_level: int
    new_level: int
    reason: str
    window_episodes: int
    success_rate: float
    safety_failure_rate: float


@dataclass
class CurriculumController:
    """按 episode 归属等级维护固定长度窗口的全局课程控制器。"""

    config: CurriculumCfg
    level: int = 0
    cooldown_remaining: int = 0
    windows: dict[int, deque[EpisodeResult]] = field(default_factory=dict)
    transitions: list[CurriculumTransition] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 <= self.level < len(self.config.levels):
            raise ValueError("初始课程等级超出配置范围")
        transition = self.config.transition
        if transition.rolling_window_episodes < transition.min_episodes_for_transition:
            raise ValueError("滚动窗口不得小于最小有效 episode 数")
        if transition.min_episodes_for_transition <= 0 or transition.cooldown_episodes < 0:
            raise ValueError("课程 episode 数和冷却期必须有效")
        for index, level_cfg in enumerate(self.config.levels):
            if len(level_cfg.target_state_probabilities) != 4 or any(
                value < 0.0 for value in level_cfg.target_state_probabilities
            ):
                raise ValueError(f"课程等级 {index} 的目标分布无效")
            if abs(sum(level_cfg.target_state_probabilities) - 1.0) > 1e-6:
                raise ValueError(f"课程等级 {index} 的目标分布必须归一化")
            if len(level_cfg.path_mode_probabilities) != 3 or any(
                value < 0.0 for value in level_cfg.path_mode_probabilities
            ):
                raise ValueError(f"课程等级 {index} 的路径模式分布无效")
            if abs(sum(level_cfg.path_mode_probabilities) - 1.0) > 1e-6:
                raise ValueError(f"课程等级 {index} 的路径模式分布必须归一化")
            if len(level_cfg.tcp_action_mask) != 6 or any(
                value not in (0.0, 1.0) for value in level_cfg.tcp_action_mask
            ):
                raise ValueError(f"课程等级 {index} 的 TCP 动作掩码必须为六维二值")
            if len(level_cfg.collision_clearance_enabled_mask) != 3 or any(
                value not in (0.0, 1.0) for value in level_cfg.collision_clearance_enabled_mask
            ):
                raise ValueError(f"课程等级 {index} 的碰撞净空掩码必须为三维二值")
            if level_cfg.collision_profile_id not in {item.identifier for item in CURRICULUM_COLLISION_PROFILES}:
                raise ValueError(f"课程等级 {index} 引用了未知碰撞配置档")
            if not 0.0 <= level_cfg.previous_level_sampling_probability <= 1.0:
                raise ValueError(f"课程等级 {index} 的上一阶段回归采样比例无效")
        for index, values in list(self.windows.items()):
            self.windows[index] = deque(values, maxlen=transition.rolling_window_episodes)

    def submit_batch(self, results: Iterable[EpisodeResult]) -> CurriculumTransition | None:
        """先按旧等级归档整批结果，再最多执行一次全局等级转换。"""

        values = tuple(results)
        if not values:
            return None
        for result in values:
            if not 0 <= result.level < len(self.config.levels):
                raise ValueError("episode 归属等级超出配置范围")
            if not 0 <= result.target_state_index < 4:
                raise ValueError("episode 目标状态索引超出配置范围")
            if not 0 <= result.path_mode_index < 3:
                raise ValueError("episode 路径模式索引超出配置范围")
            if result.collision_profile_id != self.config.levels[result.level].collision_profile_id:
                raise ValueError("episode 碰撞配置档与归属等级不一致")
            window = self.windows.setdefault(result.level, deque(maxlen=self.config.transition.rolling_window_episodes))
            window.append(
                EpisodeResult(
                    level=result.level,
                    curriculum_success=bool(result.curriculum_success and not result.safety_failure),
                    formal_parking_success=bool(result.formal_parking_success and not result.safety_failure),
                    safety_failure=bool(result.safety_failure),
                    target_state_index=int(result.target_state_index),
                    path_mode_index=int(result.path_mode_index),
                    collision_profile_id=str(result.collision_profile_id),
                )
            )
        self.cooldown_remaining = max(0, self.cooldown_remaining - len(values))
        if self.cooldown_remaining:
            return None
        return self._maybe_transition()

    def _maybe_transition(self) -> CurriculumTransition | None:
        window = self.windows.get(self.level, ())
        transition_cfg = self.config.transition
        if len(window) < transition_cfg.min_episodes_for_transition:
            return None
        success_rate = sum(result.success for result in window) / len(window)
        safety_rate = sum(result.safety_failure for result in window) / len(window)
        old_level = self.level
        if (
            old_level < len(self.config.levels) - 1
            and success_rate >= transition_cfg.promote_success_rate
            and safety_rate <= transition_cfg.max_promote_safety_failure_rate
        ):
            self.level += 1
            reason = "promoted"
        elif old_level > 0 and (
            success_rate < transition_cfg.demote_success_rate
            or safety_rate > transition_cfg.max_demote_safety_failure_rate
        ):
            self.level -= 1
            reason = "demoted"
        else:
            return None
        self.cooldown_remaining = transition_cfg.cooldown_episodes
        record = CurriculumTransition(old_level, self.level, reason, len(window), success_rate, safety_rate)
        self.transitions.append(record)
        return record

    def snapshot(self) -> dict[str, object]:
        """返回 JSON 可编码的无损状态，不嵌入训练器对象。"""

        return {
            "config_version": self.config.version,
            "observation_schema_version": self.config.observation_schema_version,
            "level": self.level,
            "cooldown_remaining": self.cooldown_remaining,
            "windows": {
                str(level): [
                    {
                        "level": item.level,
                        "curriculum_success": item.curriculum_success,
                        "formal_parking_success": item.formal_parking_success,
                        "safety_failure": item.safety_failure,
                        "target_state_index": item.target_state_index,
                        "path_mode_index": item.path_mode_index,
                        "collision_profile_id": item.collision_profile_id,
                    }
                    for item in window
                ]
                for level, window in self.windows.items()
            },
            "transitions": [
                {
                    "old_level": item.old_level,
                    "new_level": item.new_level,
                    "reason": item.reason,
                    "window_episodes": item.window_episodes,
                    "success_rate": item.success_rate,
                    "safety_failure_rate": item.safety_failure_rate,
                }
                for item in self.transitions
            ],
        }

    @classmethod
    def from_snapshot(cls, config: CurriculumCfg, snapshot: dict[str, object]) -> CurriculumController:
        if snapshot.get("config_version") != config.version:
            raise ValueError("课程状态配置版本与当前配置不一致")
        if snapshot.get("observation_schema_version") != config.observation_schema_version:
            raise ValueError("课程状态观测 schema 版本与当前配置不一致")
        raw_windows = snapshot.get("windows", {})
        if not isinstance(raw_windows, dict):
            raise ValueError("课程状态 windows 格式无效")
        windows: dict[int, deque[EpisodeResult]] = {}
        for key, entries in raw_windows.items():
            if not isinstance(entries, list):
                raise ValueError("课程状态窗口条目无效")
            try:
                windows[int(key)] = deque(
                    (EpisodeResult(**entry) for entry in entries), maxlen=config.transition.rolling_window_episodes
                )
            except (TypeError, ValueError) as error:
                raise ValueError("课程状态窗口条目与当前版本不兼容") from error
        controller = cls(
            config=config,
            level=int(snapshot["level"]),
            cooldown_remaining=int(snapshot.get("cooldown_remaining", 0)),
            windows=windows,
        )
        raw_transitions = snapshot.get("transitions", [])
        if not isinstance(raw_transitions, list):
            raise ValueError("课程状态 transitions 格式无效")
        controller.transitions = [CurriculumTransition(**entry) for entry in raw_transitions]
        return controller
