"""阶段 E 正式评估的进程无关契约与结果聚合。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


STAGE_E_EVALUATION_SCHEMA_VERSION = "stage-e-evaluation-v1"


@dataclass(frozen=True, order=True)
class EvaluationUnit:
    """一个必须在独立 SimulationApp 中完成的固定评估单元。"""

    seed: int
    target_state_index: int


def vecnormalize_path(checkpoint: Path) -> Path:
    """返回与 PPO checkpoint 成套保存的冻结 VecNormalize 路径。"""

    stem = checkpoint.stem
    periodic_match = re.fullmatch(r"(.+_model)_(\d+_steps)", stem)
    if periodic_match:
        return checkpoint.with_name(f"{periodic_match.group(1)}_vecnormalize_{periodic_match.group(2)}.pkl")
    return checkpoint.with_name(f"{stem}_vecnormalize.pkl")


def curriculum_state_path(checkpoint: Path) -> Path:
    """返回与 PPO checkpoint 成套保存的课程状态路径。"""

    return checkpoint.with_name(f"{checkpoint.stem}_curriculum.json")


def evaluation_units(seeds: Iterable[int], target_count: int) -> tuple[EvaluationUnit, ...]:
    """以固定目标、固定随机种子的笛卡尔积建立可恢复的评估矩阵。"""

    if target_count <= 0:
        raise ValueError("目标状态数量必须为正")
    unique_seeds = tuple(dict.fromkeys(seeds))
    if not unique_seeds:
        raise ValueError("至少提供一个评估随机种子")
    return tuple(EvaluationUnit(seed, target_index) for target_index in range(target_count) for seed in unique_seeds)


def validate_curriculum_snapshot(snapshot: Mapping[str, object], expected_version: str) -> None:
    """拒绝与当前任务配置不一致的 checkpoint 课程快照。"""

    if snapshot.get("config_version") != expected_version:
        raise ValueError("checkpoint 的课程状态版本与当前正式评估配置不一致")


def unit_result_filename(unit: EvaluationUnit) -> str:
    """稳定命名便于只重跑缺失或失败的评估单元。"""

    return f"seed_{unit.seed}_target_{unit.target_state_index}.json"


def validate_unit_result(
    report: Mapping[str, object],
    unit: EvaluationUnit,
    *,
    config_version: str,
    episodes_per_state: int,
    target_name: str,
) -> None:
    """确认已落盘结果可安全复用于本次聚合。"""

    expected = {
        "schema_version": STAGE_E_EVALUATION_SCHEMA_VERSION,
        "seed": unit.seed,
        "target_state_index": unit.target_state_index,
        "target_state": target_name,
        "curriculum_config_version": config_version,
        "episodes": episodes_per_state,
        "curriculum_disabled": True,
    }
    mismatched = [key for key, value in expected.items() if report.get(key) != value]
    if mismatched:
        raise ValueError(f"评估单元结果与当前请求不匹配：{', '.join(mismatched)}")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("评估单元结果缺少 metrics")


def aggregate_unit_results(
    reports: Iterable[Mapping[str, object]],
    *,
    checkpoint: Path,
    vecnormalize: Path,
    curriculum_state: Path,
    config_version: str,
    seeds: Iterable[int],
    episodes_per_state: int,
) -> dict[str, object]:
    """按目标汇总已校验的单位结果，保留每个 seed 的独立证据。"""

    by_target: dict[str, list[dict[str, object]]] = {}
    for report in reports:
        target_name = str(report["target_state"])
        by_target.setdefault(target_name, []).append(dict(report))
    for target_reports in by_target.values():
        target_reports.sort(key=lambda item: int(item["seed"]))
    return {
        "schema_version": STAGE_E_EVALUATION_SCHEMA_VERSION,
        "task": "CurriculumRL-TcpDocking-v0",
        "checkpoint": str(checkpoint.resolve()),
        "vecnormalize": str(vecnormalize.resolve()),
        "curriculum_state": str(curriculum_state.resolve()),
        "curriculum_config_version": config_version,
        "curriculum_disabled": True,
        "evaluation_observation_curriculum_level": 4,
        "evaluation_target_distribution": "per-target fixed",
        "seeds": list(dict.fromkeys(seeds)),
        "episodes_per_state_per_seed": episodes_per_state,
        "results_by_target": by_target,
    }
