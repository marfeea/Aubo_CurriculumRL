"""P4 阶段 1 评估调度器：按 policy × seed × z 生成可恢复的独立证据。"""
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
    P4_STAGE1_EVALUATION_SCHEMA_VERSION,
    Stage1EvaluationUnit,
    cluster_summary,
    stage1_evaluation_units,
    trajectory_cluster_labels,
    unit_result_filename,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", choices=("zero", "random", "ppo"), required=True, help="固定动作策略条件。")
parser.add_argument("--checkpoint", type=Path, default=None, help="PPO 条件的 checkpoint。")
parser.add_argument("--num-envs", type=int, default=4, help="并行环境数。")
parser.add_argument("--episodes-per-mode", type=int, default=32, help="每路径模式、每 seed 的完整 episode 数。")
parser.add_argument("--seeds", type=int, nargs="+", default=(7, 11, 19), help="固定评估随机种子列表。")
parser.add_argument("--json-output", type=Path, required=True, help="汇总 UTF-8 JSON 路径。")
parser.add_argument("--unit-output-dir", type=Path, default=None, help="可恢复的单位 JSON 目录。")
parser.add_argument("--rerun-all", action="store_true", help="忽略已有有效单元结果并全部重跑。")
parser.add_argument("--cluster-points", type=int, default=32, help="路径聚类前的归一化重采样点数。")
parser.add_argument(
    "--behavior-trace-stride",
    type=int,
    default=4,
    help="每个 episode 的行为轨迹每隔多少策略步保存一个样本。",
)
parser.add_argument(
    "--cluster-distance-threshold-m",
    type=float,
    default=0.08,
    help="暂定路径簇平均点距阈值；写入报告，仅用于统计，不自动决定 G4。",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs <= 0 or args_cli.episodes_per_mode <= 0 or args_cli.episodes_per_mode % args_cli.num_envs:
    parser.error("--num-envs 和 --episodes-per-mode 必须为正，且后者必须整除前者。")
if not args_cli.seeds:
    parser.error("至少提供一个 --seeds 随机种子。")
if args_cli.cluster_points < 2 or args_cli.cluster_distance_threshold_m <= 0.0:
    parser.error("路径聚类点数必须至少为 2，距离阈值必须为正。")
if args_cli.behavior_trace_stride <= 0:
    parser.error("--behavior-trace-stride 必须为正整数。")
if args_cli.policy == "ppo" and (args_cli.checkpoint is None or not args_cli.checkpoint.is_file()):
    parser.error("PPO 条件必须提供存在的 --checkpoint。")
if args_cli.policy != "ppo" and args_cli.checkpoint is not None:
    parser.error("零动作和随机动作条件不接受 --checkpoint。")


def _unit_directory() -> Path:
    return args_cli.unit_output_dir or (args_cli.json_output.parent / f"{args_cli.json_output.stem}_units")


def _validate_unit(report: Mapping[str, object], unit: Stage1EvaluationUnit) -> None:
    expected = {
        "schema_version": P4_STAGE1_EVALUATION_SCHEMA_VERSION,
        "policy": unit.policy_kind,
        "seed": unit.seed,
        "path_mode_index": unit.path_mode_index,
        "path_mode": CURRICULUM_V2_PATH_MODES[unit.path_mode_index].name,
        "episodes": args_cli.episodes_per_mode,
        "curriculum_disabled": True,
        "evaluation_curriculum_level": 0,
        "evaluation_target_state_index": 0,
        "behavior_trace_stride": args_cli.behavior_trace_stride,
    }
    mismatched = [key for key, value in expected.items() if report.get(key) != value]
    if mismatched or not isinstance(report.get("metrics"), Mapping) or not isinstance(report.get("episodes_detail"), list):
        raise ValueError(f"P4 单元结果与当前请求不匹配：{', '.join(mismatched) or 'metrics/episodes_detail'}")


def _worker_command(unit: Stage1EvaluationUnit, output_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_stage1_worker.py")),
        "--policy",
        unit.policy_kind,
        "--num-envs",
        str(args_cli.num_envs),
        "--episodes-per-mode",
        str(args_cli.episodes_per_mode),
        "--seed",
        str(unit.seed),
        "--path-mode-index",
        str(unit.path_mode_index),
        "--json-output",
        str(output_path),
        "--behavior-trace-stride",
        str(args_cli.behavior_trace_stride),
        "--device",
        str(args_cli.device),
    ]
    if args_cli.checkpoint is not None:
        command.extend(("--checkpoint", str(args_cli.checkpoint)))
    if args_cli.headless:
        command.append("--headless")
    return command


def _run_or_reuse(unit: Stage1EvaluationUnit, output_path: Path) -> Mapping[str, object]:
    if not args_cli.rerun_all and output_path.is_file():
        try:
            report = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("JSON 根节点不是对象")
            _validate_unit(report, unit)
            print(f"P4_STAGE1_REUSE_UNIT=seed={unit.seed},path={unit.path_mode_index}", flush=True)
            return report
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"P4_STAGE1_RETRY_UNIT=seed={unit.seed},path={unit.path_mode_index},reason={error}", flush=True)
    print(f"P4_STAGE1_START_UNIT=seed={unit.seed},path={unit.path_mode_index}", flush=True)
    completed = subprocess.run(_worker_command(unit, output_path), check=False)
    if completed.returncode:
        raise RuntimeError(f"P4 单元失败：seed={unit.seed}, path_mode={unit.path_mode_index}, 退出码={completed.returncode}")
    report = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"P4 子进程输出不是 JSON 对象：{output_path}")
    _validate_unit(report, unit)
    return report


def _mean(reports: list[Mapping[str, object]], key: str) -> float:
    return sum(float(report["metrics"][key]) for report in reports) / len(reports)  # type: ignore[index]


def _summarize(reports: list[Mapping[str, object]]) -> dict[str, object]:
    by_mode: dict[int, list[Mapping[str, object]]] = {index: [] for index in range(len(CURRICULUM_V2_PATH_MODES))}
    for report in reports:
        by_mode[int(report["path_mode_index"])].append(report)
    path_modes: dict[str, object] = {}
    all_successful_trajectories: list[list[list[float]]] = []
    for index, mode in enumerate(CURRICULUM_V2_PATH_MODES):
        mode_reports = by_mode[index]
        details = [episode for report in mode_reports for episode in report["episodes_detail"]]  # type: ignore[index]
        successful = [episode for episode in details if episode["curriculum_success"]]
        trajectories = [episode["tcp_trajectory_b"] for episode in successful]
        labels = trajectory_cluster_labels(
            trajectories,
            point_count=args_cli.cluster_points,
            mean_distance_threshold_m=args_cli.cluster_distance_threshold_m,
        ) if trajectories else ()
        all_successful_trajectories.extend(trajectories)
        path_modes[mode.name] = {
            "episodes": len(details),
            "curriculum_success_rate": _mean(mode_reports, "curriculum_success_rate"),
            "formal_parking_success_rate": _mean(mode_reports, "formal_parking_success_rate"),
            "safety_failure_rate": _mean(mode_reports, "safety_failure_rate"),
            "timeout_rate": _mean(mode_reports, "timeout_rate"),
            "mean_return": _mean(mode_reports, "mean_return"),
            "successful_trajectory_count": len(trajectories),
            "successful_trajectory_cluster_count": len(set(labels)),
            "successful_trajectory_cluster_samples": cluster_summary(labels),
        }
    all_labels = trajectory_cluster_labels(
        all_successful_trajectories,
        point_count=args_cli.cluster_points,
        mean_distance_threshold_m=args_cli.cluster_distance_threshold_m,
    ) if all_successful_trajectories else ()
    return {
        "schema_version": P4_STAGE1_EVALUATION_SCHEMA_VERSION,
        "policy": args_cli.policy,
        "checkpoint": str(args_cli.checkpoint.resolve()) if args_cli.checkpoint is not None else None,
        "seeds": list(dict.fromkeys(args_cli.seeds)),
        "episodes_per_mode_per_seed": args_cli.episodes_per_mode,
        "behavior_trace_stride": args_cli.behavior_trace_stride,
        "cluster": {
            "point_count": args_cli.cluster_points,
            "mean_distance_threshold_m": args_cli.cluster_distance_threshold_m,
            "status": "provisional_statistics_only_not_a_g4_decision",
        },
        "path_modes": path_modes,
        "successful_trajectory_cluster_count_across_path_modes": len(set(all_labels)),
        "successful_trajectory_cluster_samples_across_path_modes": cluster_summary(all_labels),
        "g4_decision": "pending_manual_comparison_with_zero_random_and_three_ppo_training_seeds",
    }


def main() -> dict[str, object]:
    unit_directory = _unit_directory()
    unit_directory.mkdir(parents=True, exist_ok=True)
    reports = [
        _run_or_reuse(unit, unit_directory / unit_result_filename(unit))
        for unit in stage1_evaluation_units(args_cli.policy, args_cli.seeds, len(CURRICULUM_V2_PATH_MODES))
    ]
    summary = _summarize(reports)
    args_cli.json_output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P4_STAGE1_EVALUATION=" + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    main()
