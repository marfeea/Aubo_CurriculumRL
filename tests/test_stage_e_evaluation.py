"""阶段 E 多进程正式评估调度的纯逻辑测试。"""

import sys
from pathlib import Path

import pytest

PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "source" / "CurriculumRL"
sys.path.insert(0, str(PACKAGE_SOURCE))

from CurriculumRL.logic.stage_e_evaluation import (  # noqa: E402
    STAGE_E_EVALUATION_SCHEMA_VERSION,
    EvaluationUnit,
    aggregate_unit_results,
    curriculum_state_path,
    evaluation_units,
    unit_result_filename,
    validate_curriculum_snapshot,
    validate_unit_result,
    vecnormalize_path,
)


def _report(unit: EvaluationUnit, target_name: str) -> dict[str, object]:
    return {
        "schema_version": STAGE_E_EVALUATION_SCHEMA_VERSION,
        "seed": unit.seed,
        "target_state_index": unit.target_state_index,
        "target_state": target_name,
        "curriculum_config_version": "stage-e-v1-provisional",
        "curriculum_disabled": True,
        "episodes": 8,
        "metrics": {"success_rate": 0.5},
    }


def test_checkpoint_sidecars_support_final_and_periodic_names() -> None:
    final_checkpoint = Path("run/final_model.zip")
    periodic_checkpoint = Path("run/tcp_docking_model_1024_steps.zip")
    assert vecnormalize_path(final_checkpoint) == Path("run/final_model_vecnormalize.pkl")
    assert vecnormalize_path(periodic_checkpoint) == Path("run/tcp_docking_model_vecnormalize_1024_steps.pkl")
    assert curriculum_state_path(final_checkpoint) == Path("run/final_model_curriculum.json")


def test_evaluation_matrix_is_unique_and_result_files_are_stable() -> None:
    units = evaluation_units((7, 11, 7), target_count=2)
    assert units == (
        EvaluationUnit(7, 0),
        EvaluationUnit(11, 0),
        EvaluationUnit(7, 1),
        EvaluationUnit(11, 1),
    )
    assert unit_result_filename(EvaluationUnit(11, 3)) == "seed_11_target_3.json"


def test_snapshot_and_worker_report_reject_version_or_unit_mismatch() -> None:
    with pytest.raises(ValueError, match="版本"):
        validate_curriculum_snapshot({"config_version": "old"}, "stage-e-v1-provisional")
    unit = EvaluationUnit(7, 0)
    report = _report(unit, "target_a")
    validate_unit_result(
        report,
        unit,
        config_version="stage-e-v1-provisional",
        episodes_per_state=8,
        target_name="target_a",
    )
    report["seed"] = 11
    with pytest.raises(ValueError, match="seed"):
        validate_unit_result(
            report,
            unit,
            config_version="stage-e-v1-provisional",
            episodes_per_state=8,
            target_name="target_a",
        )


def test_aggregate_preserves_partial_results_by_target() -> None:
    reports = [_report(EvaluationUnit(11, 0), "target_a"), _report(EvaluationUnit(7, 0), "target_a")]
    summary = aggregate_unit_results(
        reports,
        checkpoint=Path("run/model.zip"),
        vecnormalize=Path("run/model_vecnormalize.pkl"),
        curriculum_state=Path("run/model_curriculum.json"),
        config_version="stage-e-v1-provisional",
        seeds=(7, 11),
        episodes_per_state=8,
    )
    assert [item["seed"] for item in summary["results_by_target"]["target_a"]] == [7, 11]
