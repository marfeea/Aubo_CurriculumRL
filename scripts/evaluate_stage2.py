"""P5 阶段 2 调度器：同时评估 L1 位姿能力和 L0 固定回归。"""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from _bootstrap import add_package_source

add_package_source()

from isaaclab.app import AppLauncher  # noqa: E402

from CurriculumRL.configs.curriculum import CURRICULUM_V2_PATH_MODES  # noqa: E402
from CurriculumRL.logic.stage_p4_evaluation import (  # noqa: E402
    cluster_summary,
    trajectory_cluster_labels,
)
from CurriculumRL.logic.stage_p5_evaluation import (  # noqa: E402
    P5_STAGE2_EVALUATION_SCHEMA_VERSION,
    Stage2EvaluationUnit,
    capability_name,
    stage2_evaluation_units,
    unit_result_filename,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", choices=("zero", "random", "ppo"), required=True)
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--episodes-per-condition", type=int, default=32)
parser.add_argument("--seeds", type=int, nargs="+", default=(7, 11, 19))
parser.add_argument(
    "--evaluation-levels",
    type=int,
    nargs="+",
    choices=(0, 1),
    default=(1, 0),
    help="默认同时运行 L1 位姿能力与 L0 固定回归。",
)
parser.add_argument("--json-output", type=Path, required=True)
parser.add_argument("--unit-output-dir", type=Path, default=None)
parser.add_argument("--rerun-all", action="store_true")
parser.add_argument("--cluster-points", type=int, default=32)
parser.add_argument("--cluster-distance-threshold-m", type=float, default=0.08)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_PROTECTION_NAMES = ("singular", "delta_limited", "velocity_limited", "position_limited")
_ACTION_GROUP_NAMES = ("translation", "rotation")

if (
    args_cli.num_envs <= 0
    or args_cli.episodes_per_condition <= 0
    or args_cli.episodes_per_condition % args_cli.num_envs
):
    parser.error("--num-envs 和 --episodes-per-condition 必须为正，且后者必须整除前者。")
if not args_cli.seeds:
    parser.error("至少提供一个 --seeds 随机种子。")
if args_cli.cluster_points < 2 or args_cli.cluster_distance_threshold_m <= 0.0:
    parser.error("路径聚类点数必须至少为 2，距离阈值必须为正。")
if args_cli.policy == "ppo" and (
    args_cli.checkpoint is None or not args_cli.checkpoint.is_file()
):
    parser.error("PPO 条件必须提供存在的 --checkpoint。")
if args_cli.policy != "ppo" and args_cli.checkpoint is not None:
    parser.error("零动作和随机动作条件不接受 --checkpoint。")


def _unit_directory() -> Path:
    return args_cli.unit_output_dir or (
        args_cli.json_output.parent / f"{args_cli.json_output.stem}_units"
    )


def _validate_unit(
    report: Mapping[str, object],
    unit: Stage2EvaluationUnit,
) -> None:
    expected = {
        "schema_version": P5_STAGE2_EVALUATION_SCHEMA_VERSION,
        "policy": unit.policy_kind,
        "seed": unit.seed,
        "path_mode_index": unit.path_mode_index,
        "path_mode": CURRICULUM_V2_PATH_MODES[unit.path_mode_index].name,
        "episodes": args_cli.episodes_per_condition,
        "curriculum_disabled": True,
        "evaluation_curriculum_level": unit.evaluation_level,
        "evaluation_target_state_index": 0,
    }
    mismatched = [
        key for key, value in expected.items() if report.get(key) != value
    ]
    if (
        mismatched
        or not isinstance(report.get("metrics"), Mapping)
        or not isinstance(report.get("episodes_detail"), list)
    ):
        raise ValueError(
            "P5 单元结果与当前请求不匹配："
            + (", ".join(mismatched) or "metrics/episodes_detail")
        )


def _worker_command(
    unit: Stage2EvaluationUnit,
    output_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_stage2_worker.py")),
        "--policy",
        unit.policy_kind,
        "--num-envs",
        str(args_cli.num_envs),
        "--episodes-per-condition",
        str(args_cli.episodes_per_condition),
        "--seed",
        str(unit.seed),
        "--path-mode-index",
        str(unit.path_mode_index),
        "--evaluation-level",
        str(unit.evaluation_level),
        "--json-output",
        str(output_path),
        "--device",
        str(args_cli.device),
    ]
    if args_cli.checkpoint is not None:
        command.extend(("--checkpoint", str(args_cli.checkpoint)))
    if args_cli.headless:
        command.append("--headless")
    return command


def _run_or_reuse(
    unit: Stage2EvaluationUnit,
    output_path: Path,
) -> Mapping[str, object]:
    label = (
        f"level={unit.evaluation_level},seed={unit.seed},"
        f"path={unit.path_mode_index}"
    )
    if not args_cli.rerun_all and output_path.is_file():
        try:
            report = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("JSON 根节点不是对象")
            _validate_unit(report, unit)
            print(f"P5_STAGE2_REUSE_UNIT={label}", flush=True)
            return report
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"P5_STAGE2_RETRY_UNIT={label},reason={error}", flush=True)
    print(f"P5_STAGE2_START_UNIT={label}", flush=True)
    completed = subprocess.run(
        _worker_command(unit, output_path),
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"P5 单元失败：{label}, 退出码={completed.returncode}")
    report = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"P5 子进程输出不是 JSON 对象：{output_path}")
    _validate_unit(report, unit)
    return report


def _mean(reports: list[Mapping[str, object]], key: str) -> float:
    return sum(
        float(report["metrics"][key]) for report in reports  # type: ignore[index]
    ) / len(reports)


def _max_nested(
    reports: list[Mapping[str, object]],
    group: str,
    key: str,
) -> float:
    return max(
        float(report["metrics"][group][key])  # type: ignore[index]
        for report in reports
    )


def _summarize_capability(
    reports: list[Mapping[str, object]],
    evaluation_level: int,
) -> dict[str, object]:
    selected = [
        report
        for report in reports
        if report["evaluation_curriculum_level"] == evaluation_level
    ]
    path_modes: dict[str, object] = {}
    all_successful_trajectories: list[list[list[float]]] = []
    for index, mode in enumerate(CURRICULUM_V2_PATH_MODES):
        mode_reports = [
            report for report in selected if report["path_mode_index"] == index
        ]
        details = [
            episode
            for report in mode_reports
            for episode in report["episodes_detail"]  # type: ignore[index]
        ]
        successful = [
            episode for episode in details if episode["curriculum_success"]
        ]
        trajectories = [
            episode["tcp_trajectory_b"] for episode in successful
        ]
        labels = (
            trajectory_cluster_labels(
                trajectories,
                point_count=args_cli.cluster_points,
                mean_distance_threshold_m=args_cli.cluster_distance_threshold_m,
            )
            if trajectories
            else ()
        )
        all_successful_trajectories.extend(trajectories)
        path_modes[mode.name] = {
            "episodes": len(details),
            "curriculum_success_rate": _mean(
                mode_reports,
                "curriculum_success_rate",
            ),
            "formal_parking_success_rate": _mean(
                mode_reports,
                "formal_parking_success_rate",
            ),
            "safety_failure_rate": _mean(
                mode_reports,
                "safety_failure_rate",
            ),
            "timeout_rate": _mean(mode_reports, "timeout_rate"),
            "mean_return": _mean(mode_reports, "mean_return"),
            "mean_final_position_error_m": _mean(
                mode_reports,
                "mean_final_position_error_m",
            ),
            "mean_position_error_improvement_m": _mean(
                mode_reports,
                "mean_position_error_improvement_m",
            ),
            "mean_final_orientation_error_rad": _mean(
                mode_reports,
                "mean_final_orientation_error_rad",
            ),
            "mean_orientation_error_improvement_rad": _mean(
                mode_reports,
                "mean_orientation_error_improvement_rad",
            ),
            "max_controller_protection_rate_per_policy_step": {
                name: _max_nested(
                    mode_reports,
                    "controller_protection_rate_per_policy_step",
                    name,
                )
                for name in _PROTECTION_NAMES
            },
            "max_controller_protection_consecutive_policy_steps": {
                name: int(
                    _max_nested(
                        mode_reports,
                        "controller_protection_max_consecutive_policy_steps",
                        name,
                    )
                )
                for name in _PROTECTION_NAMES
            },
            "max_action_component_saturation_rate": {
                name: _max_nested(
                    mode_reports,
                    "action_component_saturation_rate",
                    name,
                )
                for name in _ACTION_GROUP_NAMES
            },
            "max_joint_position_limit_violation_rad": max(
                float(
                    report["metrics"][  # type: ignore[index]
                        "max_joint_position_limit_violation_rad"
                    ]
                )
                for report in mode_reports
            ),
            "successful_trajectory_count": len(trajectories),
            "successful_trajectory_cluster_count": len(set(labels)),
            "successful_trajectory_cluster_samples": cluster_summary(labels),
        }
    all_labels = (
        trajectory_cluster_labels(
            all_successful_trajectories,
            point_count=args_cli.cluster_points,
            mean_distance_threshold_m=args_cli.cluster_distance_threshold_m,
        )
        if all_successful_trajectories
        else ()
    )
    return {
        "evaluation_curriculum_level": evaluation_level,
        "path_modes": path_modes,
        "successful_trajectory_cluster_count_across_path_modes": len(
            set(all_labels)
        ),
        "successful_trajectory_cluster_samples_across_path_modes": (
            cluster_summary(all_labels)
        ),
    }


def _summarize(reports: list[Mapping[str, object]]) -> dict[str, object]:
    levels = tuple(dict.fromkeys(args_cli.evaluation_levels))
    return {
        "schema_version": P5_STAGE2_EVALUATION_SCHEMA_VERSION,
        "policy": args_cli.policy,
        "checkpoint": (
            str(args_cli.checkpoint.resolve())
            if args_cli.checkpoint is not None
            else None
        ),
        "seeds": list(dict.fromkeys(args_cli.seeds)),
        "episodes_per_condition_per_seed": args_cli.episodes_per_condition,
        "cluster": {
            "point_count": args_cli.cluster_points,
            "mean_distance_threshold_m": args_cli.cluster_distance_threshold_m,
            "status": "provisional_statistics_only_not_a_g5_decision",
        },
        "capabilities": {
            capability_name(level): _summarize_capability(reports, level)
            for level in levels
        },
        "g5_decision": (
            "pending_manual_comparison_with_stage1_checkpoint_baseline_"
            "and_zero_random_and_three_stage2_ppo_training_seeds"
        ),
    }


def main() -> dict[str, object]:
    unit_directory = _unit_directory()
    unit_directory.mkdir(parents=True, exist_ok=True)
    units = stage2_evaluation_units(
        args_cli.policy,
        args_cli.seeds,
        len(CURRICULUM_V2_PATH_MODES),
        args_cli.evaluation_levels,
    )
    reports = [
        _run_or_reuse(unit, unit_directory / unit_result_filename(unit))
        for unit in units
    ]
    summary = _summarize(reports)
    args_cli.json_output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.json_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "P5_STAGE2_EVALUATION="
        + json.dumps(summary, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    main()
