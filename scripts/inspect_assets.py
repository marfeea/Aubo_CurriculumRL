"""检查阶段 A 外部资产、递归 USD 依赖和机器人 prim 契约。"""

from __future__ import annotations

import argparse
import json
import sys

from _bootstrap import add_package_source

add_package_source()

from CurriculumRL.configs.assets import ASSET_ROOT_ENV_VAR, AssetRootError, resolve_asset_root  # noqa: E402
from CurriculumRL.runtime.asset_inspection import check_asset_files  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", help=f"覆盖资产根；默认读取 {ASSET_ROOT_ENV_VAR} 或工作区相邻 Asset")
    parser.add_argument(
        "--filesystem-only",
        action="store_true",
        help="仅检查清单文件；不启动 Kit，因此不验证 USD 依赖和 prim 契约",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        root = resolve_asset_root(args.asset_root)
        file_results = check_asset_files(root)
    except AssetRootError as error:
        print(f"[资产根错误] {error}", file=sys.stderr)
        return 2

    if args.filesystem_only:
        report = {
            "asset_root": str(root),
            "mode": "filesystem-only",
            "assets": [
                {"key": spec.key, "path": str(path), "required": spec.required, "exists": exists}
                for spec, path, exists in file_results
            ],
        }
        _print_report(report, args.json)
        return int(any(spec.required and not exists for spec, _, exists in file_results))

    try:
        from isaacsim import SimulationApp
    except ModuleNotFoundError as error:
        print(
            f"[运行环境错误] 完整检查必须使用 Isaac Sim Python；可先用 --filesystem-only。\n原始错误：{error}",
            file=sys.stderr,
        )
        return 2

    app = SimulationApp({"headless": True})
    try:
        from CurriculumRL.runtime.asset_inspection import inspect_asset_stages

        results = inspect_asset_stages(root)
        report = {
            "asset_root": str(root),
            "mode": "usd-stage",
            "assets": [result.to_dict() for result in results],
        }
        _print_report(report, args.json)
        return int(not all(result.passed for result in results))
    finally:
        app.close()


def _print_report(report: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    print(f"资产根：{report['asset_root']}", flush=True)
    print(f"检查模式：{report['mode']}", flush=True)
    for item in report["assets"]:
        status = "通过" if item.get("passed", item["exists"]) else "失败"
        print(f"- [{status}] {item['key']}: {item['path']}", flush=True)
        for error in item.get("errors", ()):
            print(f"  错误：{error}", flush=True)
        for path in item.get("unresolved_dependencies", ()):
            print(f"  未解析依赖：{path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
