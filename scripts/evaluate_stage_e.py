"""阶段 E 正式评估调度器：每个固定评估单元由独立子进程执行。"""
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

from CurriculumRL.configs.curriculum import CURRICULUM_CONFIG_VERSION  # noqa: E402
from CurriculumRL.configs.task import TARGET_STATES  # noqa: E402
from CurriculumRL.logic.stage_e_evaluation import (  # noqa: E402
    EvaluationUnit,
    aggregate_unit_results,
    curriculum_state_path,
    evaluation_units,
    unit_result_filename,
    validate_curriculum_snapshot,
    validate_unit_result,
    vecnormalize_path,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True, help="阶段 E PPO checkpoint 路径。")
parser.add_argument("--num-envs", type=int, default=4, help="并行环境数；必须整除每状态 episode 数。")
parser.add_argument("--episodes-per-state", type=int, default=32, help="每个目标状态、每个随机种子的完整 episode 数。")
parser.add_argument("--seeds", type=int, nargs="+", default=(7, 11, 19), help="固定评估随机种子列表。")
parser.add_argument("--json-output", type=Path, default=None, help="可选 UTF-8 汇总 JSON 结果路径。")
parser.add_argument("--unit-output-dir", type=Path, default=None, help="评估单元 JSON 目录；默认位于汇总或 checkpoint 旁。")
parser.add_argument("--rerun-all", action="store_true", help="忽略已有的有效单元结果并全部重跑。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.checkpoint.is_file():
    parser.error(f"checkpoint 不存在：{args_cli.checkpoint}")
if args_cli.num_envs <= 0 or args_cli.episodes_per_state <= 0 or args_cli.episodes_per_state % args_cli.num_envs:
    parser.error("--num-envs 和 --episodes-per-state 必须为正，且后者必须整除前者。")
if not args_cli.seeds:
    parser.error("至少提供一个 --seeds 随机种子。")


def _unit_output_dir() -> Path:
    if args_cli.unit_output_dir is not None:
        return args_cli.unit_output_dir
    if args_cli.json_output is not None:
        return args_cli.json_output.parent / f"{args_cli.json_output.stem}_units"
    return args_cli.checkpoint.parent / f"{args_cli.checkpoint.stem}_stage_e_units"


def _load_reusable_unit(unit: EvaluationUnit, output_path: Path) -> Mapping[str, object] | None:
    if args_cli.rerun_all or not output_path.is_file():
        return None
    try:
        report = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("JSON 根节点不是对象")
        validate_unit_result(
            report,
            unit,
            config_version=CURRICULUM_CONFIG_VERSION,
            episodes_per_state=args_cli.episodes_per_state,
            target_name=TARGET_STATES[unit.target_state_index].name,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"STAGE_E_RETRY_UNIT=seed={unit.seed},target={unit.target_state_index},reason={error}", flush=True)
        return None
    print(f"STAGE_E_REUSE_UNIT=seed={unit.seed},target={unit.target_state_index},path={output_path}", flush=True)
    return report


def _worker_command(unit: EvaluationUnit, output_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_stage_e_worker.py")),
        "--checkpoint",
        str(args_cli.checkpoint),
        "--num-envs",
        str(args_cli.num_envs),
        "--episodes-per-state",
        str(args_cli.episodes_per_state),
        "--seed",
        str(unit.seed),
        "--target-state-index",
        str(unit.target_state_index),
        "--json-output",
        str(output_path),
        "--device",
        str(args_cli.device),
    ]
    if args_cli.headless:
        command.append("--headless")
    return command


def _run_unit(unit: EvaluationUnit, output_path: Path) -> Mapping[str, object]:
    print(f"STAGE_E_START_UNIT=seed={unit.seed},target={unit.target_state_index}", flush=True)
    completed = subprocess.run(_worker_command(unit, output_path), check=False)
    if completed.returncode:
        failure_path = output_path.with_suffix(".failure.json")
        failure_path.write_text(
            json.dumps(
                {
                    "seed": unit.seed,
                    "target_state_index": unit.target_state_index,
                    "returncode": completed.returncode,
                    "command": _worker_command(unit, output_path),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"评估单元失败：seed={unit.seed}, target_state_index={unit.target_state_index}, "
            f"退出码={completed.returncode}；保留的失败证据：{failure_path}"
        )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"评估子进程输出不是 JSON 对象：{output_path}")
    validate_unit_result(
        report,
        unit,
        config_version=CURRICULUM_CONFIG_VERSION,
        episodes_per_state=args_cli.episodes_per_state,
        target_name=TARGET_STATES[unit.target_state_index].name,
    )
    return report


def main() -> dict[str, object]:
    vecnormalize = vecnormalize_path(args_cli.checkpoint)
    curriculum_state = curriculum_state_path(args_cli.checkpoint)
    if not vecnormalize.is_file():
        raise FileNotFoundError(f"正式评估缺少冻结的 VecNormalize 状态：{vecnormalize}")
    if not curriculum_state.is_file():
        raise FileNotFoundError(f"正式评估缺少与 checkpoint 关联的课程状态：{curriculum_state}")
    validate_curriculum_snapshot(json.loads(curriculum_state.read_text(encoding="utf-8")), CURRICULUM_CONFIG_VERSION)

    unit_directory = _unit_output_dir()
    unit_directory.mkdir(parents=True, exist_ok=True)
    reports: list[Mapping[str, object]] = []
    for unit in evaluation_units(args_cli.seeds, len(TARGET_STATES)):
        output_path = unit_directory / unit_result_filename(unit)
        report = _load_reusable_unit(unit, output_path)
        if report is None:
            report = _run_unit(unit, output_path)
        reports.append(report)
    summary = aggregate_unit_results(
        reports,
        checkpoint=args_cli.checkpoint,
        vecnormalize=vecnormalize,
        curriculum_state=curriculum_state,
        config_version=CURRICULUM_CONFIG_VERSION,
        seeds=args_cli.seeds,
        episodes_per_state=args_cli.episodes_per_state,
    )
    print("STAGE_E_EVALUATION=" + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if args_cli.json_output is not None:
        args_cli.json_output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
