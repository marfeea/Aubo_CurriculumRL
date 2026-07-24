"""用三路 CameraSensor 与 USD CollisionAPI 输出 P0 场景语义审计 JSON。"""

from __future__ import annotations

import argparse
import json
import os
import site
import sys
from pathlib import Path
from typing import Any

from _bootstrap import add_package_source

_user_site = Path(site.getusersitepackages()).resolve()
sys.path[:] = [path for path in sys.path if Path(path).resolve() != _user_site]

add_package_source()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--warmup-steps", type=int, default=8)
parser.add_argument("--max-instances-per-camera", type=int, default=64)
parser.add_argument("--camera-index", type=int, choices=range(3), help="只采集指定相机；用于顺序低显存审计。")
parser.add_argument("--output", type=Path, help="可选的审计 JSON 输出路径；默认仅打印到标准输出。")
parser.add_argument("--usd-only", action="store_true", help="仅查询工作站 USD 的 CollisionAPI；不创建仿真或 CameraSensor。")
parser.add_argument("--launch-only", action="store_true", help="仅启动带 CameraSensor 支持的 Kit，用于隔离渲染启动故障。")
parser.add_argument(
    "--kit-cache-dir",
    type=Path,
    default=Path("data/isaac_kit_cache"),
    help="Kit 用户配置与 OptiX 缓存目录；默认位于已忽略的 data/ 下。",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = not args_cli.usd_only
kit_cache_dir = args_cli.kit_cache_dir.resolve()
kit_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["OPTIX_CACHE_PATH"] = str(kit_cache_dir / "optix")
user_config_path = kit_cache_dir / "user.config.json"
args_cli.kit_args = " ".join(
    argument for argument in (args_cli.kit_args, f"--/app/userConfigPath={user_config_path}") if argument
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if args_cli.launch_only:
    launch_report = json.dumps({"camera_render_app_launched": True}, ensure_ascii=False)
    if args_cli.output:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(launch_report + "\n", encoding="utf-8")
    print("CAMERA_RENDER_APP_LAUNCHED", flush=True)
    simulation_app.close()
    raise SystemExit(0)

import torch  # noqa: E402
from CurriculumRL.configs.training import SIMULATION_DT_S  # noqa: E402
from CurriculumRL.configs.assets import asset_by_key, asset_path  # noqa: E402
from CurriculumRL.configs.vision_diagnostics import (  # noqa: E402
    DIAGNOSTIC_CAMERA_DATA_TYPES,
    DIAGNOSTIC_CAMERA_POSES,
)
from CurriculumRL.tasks.tcp_docking.vision_scene_cfg import DIAGNOSTIC_VISION_SCENE_CFGS  # noqa: E402
from isaaclab import sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402
import omni.usd  # noqa: E402


def _json_safe(value: Any) -> Any:
    """将 Replicator 的 info 字段收敛为可写入审计快照的值。"""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _prim_paths(value: Any) -> set[str]:
    """从实例分割 metadata 中提取可回查的 USD prim 路径。"""

    if isinstance(value, str):
        return {value} if value.startswith("/") else set()
    if isinstance(value, dict):
        return set().union(*(_prim_paths(item) for item in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_prim_paths(item) for item in value)) if value else set()
    return set()


def _bounds(points: torch.Tensor) -> dict[str, list[float]] | None:
    if points.numel() == 0:
        return None
    return {
        "min_e_m": points.amin(dim=0).detach().cpu().tolist(),
        "max_e_m": points.amax(dim=0).detach().cpu().tolist(),
    }


def _camera_instance_report(scene: InteractiveScene, camera_index: int) -> dict[str, Any]:
    pose_cfg = DIAGNOSTIC_CAMERA_POSES[camera_index]
    camera = scene[pose_cfg.scene_name]
    outputs = camera.data.output
    missing = set(DIAGNOSTIC_CAMERA_DATA_TYPES).difference(outputs)
    if missing:
        raise RuntimeError(f"{pose_cfg.scene_name}: 缺少输出 {sorted(missing)}")

    depth = outputs["distance_to_image_plane"][0]
    if depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    instance_ids = outputs["instance_id_segmentation_fast"][0]
    if instance_ids.shape[-1] == 1:
        instance_ids = instance_ids.squeeze(-1)
    if depth.shape != instance_ids.shape:
        raise RuntimeError(f"{pose_cfg.scene_name}: 深度与实例 ID 形状不一致：{depth.shape} != {instance_ids.shape}")

    finite_depth = torch.isfinite(depth) & (depth > 0.0)
    if not bool(finite_depth.any()):
        raise RuntimeError(f"{pose_cfg.scene_name}: 没有有限正深度")
    point_cloud_w = math_utils.transform_points(
        math_utils.unproject_depth(depth, camera.data.intrinsic_matrices[0]),
        camera.data.pos_w[0],
        camera.data.quat_w_ros[0],
    )
    point_cloud_e = point_cloud_w - scene.env_origins[0]
    flattened_depth = depth.transpose(0, 1).reshape(-1)
    flattened_ids = instance_ids.transpose(0, 1).reshape(-1)
    valid_points = torch.isfinite(point_cloud_e).all(dim=1) & (flattened_depth > 0.0)

    info = camera.data.info[0].get("instance_id_segmentation_fast", {})
    labels = info.get("idToLabels", info) if isinstance(info, dict) else {}
    instances: list[dict[str, Any]] = []
    for instance_id in torch.unique(flattened_ids).detach().cpu().tolist():
        if len(instances) >= args_cli.max_instances_per_camera:
            break
        instance_mask = valid_points & (flattened_ids == instance_id)
        count = int(instance_mask.sum().item())
        if count == 0:
            continue
        metadata = labels.get(str(instance_id), labels.get(instance_id, {})) if isinstance(labels, dict) else {}
        instances.append(
            {
                "instance_id": int(instance_id),
                "pixel_count": count,
                "prim_paths": sorted(_prim_paths(metadata)),
                "spatial_range_e": _bounds(point_cloud_e[instance_mask]),
            }
        )

    return {
        "scene_name": pose_cfg.scene_name,
        "source_reference": pose_cfg.source_reference,
        "configured_position_e_m": list(pose_cfg.position_e),
        "actual_position_w_m": camera.data.pos_w[0].detach().cpu().tolist(),
        "actual_quaternion_wxyz": camera.data.quat_w_world[0].detach().cpu().tolist(),
        "intrinsic_matrix": camera.data.intrinsic_matrices[0].detach().cpu().tolist(),
        "depth_shape": list(depth.shape),
        "finite_positive_depth_ratio": float(finite_depth.float().mean().item()),
        "instance_mapping": _json_safe(info),
        "visible_instances": instances,
    }


def _collision_prims() -> list[dict[str, Any]]:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage 未初始化")
    root = stage.GetPrimAtPath("/World/envs/env_0/station/static/workstation")
    if not root.IsValid():
        raise RuntimeError("未找到工作站 prim：/World/envs/env_0/station/static/workstation")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    result: list[dict[str, Any]] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        result.append(
            {
                "prim_path": str(prim.GetPath()),
                "type_name": prim.GetTypeName(),
                "applied_schemas": list(prim.GetAppliedSchemas()),
                "world_bounds_m": {
                    "min_w_m": list(bounds.GetMin()),
                    "max_w_m": list(bounds.GetMax()),
                },
            }
        )
    return result


def _workstation_asset_collision_prims() -> list[dict[str, Any]]:
    """直接打开工作站 USD，避免把渲染初始化问题混入 CollisionAPI 事实查询。"""

    path = asset_path(asset_by_key("workstation"))
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"无法打开工作站 USD：{path}")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    result: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        result.append(
            {
                "prim_path": str(prim.GetPath()),
                "type_name": prim.GetTypeName(),
                "applied_schemas": list(prim.GetAppliedSchemas()),
                "asset_bounds_m": {
                    "min_asset_m": list(bounds.GetMin()),
                    "max_asset_m": list(bounds.GetMax()),
                },
            }
        )
    return result


def main() -> int:
    if args_cli.warmup_steps < 1:
        raise ValueError("--warmup-steps 必须至少为 1")
    if args_cli.max_instances_per_camera < 1:
        raise ValueError("--max-instances-per-camera 必须至少为 1")
    if not args_cli.usd_only and args_cli.camera_index is None:
        raise ValueError("相机审计必须显式指定 --camera-index；请依次运行 0、1、2。")

    if args_cli.usd_only:
        report = {
            "schema_version": "p0-workstation-usd-audit-v1",
            "workstation_asset_collision_prims": _workstation_asset_collision_prims(),
        }
    else:
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIMULATION_DT_S, device=args_cli.device))
        camera_index = args_cli.camera_index
        scene = InteractiveScene(
            DIAGNOSTIC_VISION_SCENE_CFGS[camera_index](num_envs=1, env_spacing=4.0, replicate_physics=False)
        )
        sim.reset()
        scene.reset()
        for _ in range(args_cli.warmup_steps):
            scene.write_data_to_sim()
            sim.step()
            scene.update(SIMULATION_DT_S)

        report = {
            "schema_version": "p0-scene-semantics-v1",
            "camera_data_types": list(DIAGNOSTIC_CAMERA_DATA_TYPES),
            "cameras": [_camera_instance_report(scene, camera_index)],
            "workstation_collision_prims": _collision_prims(),
        }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args_cli.output:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(serialized + "\n", encoding="utf-8")
    print("SCENE_SEMANTICS_AUDIT=" + serialized, flush=True)
    if not args_cli.usd_only:
        sim.clear_instance()
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
