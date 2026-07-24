"""启动单环境三相机诊断场景并验证真实视觉传感器张量。"""

from __future__ import annotations

import argparse
import json

from _bootstrap import add_package_source

add_package_source()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--warmup-steps", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from CurriculumRL.configs.training import SIMULATION_DT_S  # noqa: E402
from CurriculumRL.configs.vision_diagnostics import (  # noqa: E402
    DIAGNOSTIC_CAMERA_DATA_TYPES,
    DIAGNOSTIC_CAMERA_POSES,
)
from CurriculumRL.tasks.tcp_docking.vision_scene_cfg import TcpDockingVisionSceneCfg  # noqa: E402
from isaaclab import sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402


def _validate_camera(scene: InteractiveScene, camera_index: int) -> dict[str, object]:
    pose_cfg = DIAGNOSTIC_CAMERA_POSES[camera_index]
    camera = scene[pose_cfg.scene_name]
    outputs = camera.data.output
    missing = set(DIAGNOSTIC_CAMERA_DATA_TYPES).difference(outputs)
    if missing:
        raise RuntimeError(f"{pose_cfg.scene_name}: 缺少视觉输出 {sorted(missing)}")

    rgb = outputs["rgb"]
    depth = outputs["distance_to_image_plane"]
    instance_ids = outputs["instance_id_segmentation_fast"]
    if rgb.shape[:3] != (1, 480, 640):
        raise RuntimeError(f"{pose_cfg.scene_name}: RGB 形状异常 {tuple(rgb.shape)}")
    if depth.shape[:3] != (1, 480, 640):
        raise RuntimeError(f"{pose_cfg.scene_name}: 深度形状异常 {tuple(depth.shape)}")
    finite_depth = torch.isfinite(depth) & (depth > 0.0)
    finite_depth_ratio = float(finite_depth.float().mean().item())
    if finite_depth_ratio <= 0.0:
        raise RuntimeError(f"{pose_cfg.scene_name}: 没有读取到有限正深度")

    unique_instance_ids = int(torch.unique(instance_ids).numel())
    if unique_instance_ids < 2:
        raise RuntimeError(f"{pose_cfg.scene_name}: 只读取到 {unique_instance_ids} 个实例 ID")

    actual_position = camera.data.pos_w[0].detach().cpu()
    expected_position = torch.tensor(pose_cfg.position_e, dtype=actual_position.dtype)
    if not torch.allclose(actual_position, expected_position, atol=1.0e-4, rtol=0.0):
        raise RuntimeError(
            f"{pose_cfg.scene_name}: 实际位置 {actual_position.tolist()} 与配置 {list(pose_cfg.position_e)} 不一致"
        )

    info = camera.data.info[0]
    segmentation_info = info.get("instance_id_segmentation_fast", {})
    return {
        "scene_name": pose_cfg.scene_name,
        "source_reference": pose_cfg.source_reference,
        "configured_position_e": list(pose_cfg.position_e),
        "actual_position_w": actual_position.tolist(),
        "actual_quaternion_world_wxyz": camera.data.quat_w_world[0].detach().cpu().tolist(),
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "finite_positive_depth_ratio": finite_depth_ratio,
        "unique_instance_ids": unique_instance_ids,
        "instance_info_keys": sorted(segmentation_info),
        "instance_info": segmentation_info,
    }


def main() -> int:
    if args_cli.warmup_steps < 1:
        raise ValueError("--warmup-steps 必须至少为 1")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIMULATION_DT_S, device=args_cli.device))
    scene = InteractiveScene(TcpDockingVisionSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False))
    sim.reset()
    scene.reset()
    for _ in range(args_cli.warmup_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(SIMULATION_DT_S)

    report = [_validate_camera(scene, index) for index in range(len(DIAGNOSTIC_CAMERA_POSES))]
    print("VISION_SENSOR_SMOKE=" + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    sim.clear_instance()
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
