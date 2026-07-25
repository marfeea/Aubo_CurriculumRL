"""P5 阶段 2 位姿能力与阶段 1 回归的固定评估矩阵。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .stage_p4_evaluation import P4_STAGE1_POLICY_KINDS


P5_STAGE2_EVALUATION_SCHEMA_VERSION = "p5-stage2-evaluation-v1"
P5_EVALUATION_LEVELS = (1, 0)


@dataclass(frozen=True, order=True)
class Stage2EvaluationUnit:
    """一个固定能力等级、策略、随机种子和路径模式的 SimulationApp 单元。"""

    policy_kind: str
    seed: int
    path_mode_index: int
    evaluation_level: int


def stage2_evaluation_units(
    policy_kind: str,
    seeds: Iterable[int],
    path_mode_count: int,
    evaluation_levels: Iterable[int] = P5_EVALUATION_LEVELS,
) -> tuple[Stage2EvaluationUnit, ...]:
    """建立 L1 位姿评估与 L0 固定回归的共同矩阵。"""

    if policy_kind not in P4_STAGE1_POLICY_KINDS:
        raise ValueError(f"未知 P5 策略条件：{policy_kind}")
    if path_mode_count <= 0:
        raise ValueError("路径模式数量必须为正")
    unique_seeds = tuple(dict.fromkeys(seeds))
    if not unique_seeds:
        raise ValueError("至少提供一个固定随机种子")
    unique_levels = tuple(dict.fromkeys(evaluation_levels))
    if not unique_levels or any(level not in P5_EVALUATION_LEVELS for level in unique_levels):
        raise ValueError("P5 固定评估等级只能是 L1 和 L0")
    return tuple(
        Stage2EvaluationUnit(policy_kind, seed, mode, level)
        for level in unique_levels
        for mode in range(path_mode_count)
        for seed in unique_seeds
    )


def unit_result_filename(unit: Stage2EvaluationUnit) -> str:
    return (
        f"{unit.policy_kind}_level_{unit.evaluation_level}_"
        f"seed_{unit.seed}_path_{unit.path_mode_index}.json"
    )


def capability_name(evaluation_level: int) -> str:
    if evaluation_level == 1:
        return "stage2_pose"
    if evaluation_level == 0:
        return "stage1_regression"
    raise ValueError("P5 固定评估等级只能是 L1 和 L0")
