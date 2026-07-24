"""带三路 CameraSensor 的单环境视觉语义诊断场景。"""

from __future__ import annotations

from isaaclab import sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from ...configs.assets import asset_by_key, asset_path
from ...configs.scene import WORKSTATION_POSE_E
from ...configs.task import TARGET_STATES
from ...configs.vision_diagnostics import (
    DIAGNOSTIC_CAMERA_DATA_TYPES,
    DIAGNOSTIC_CAMERA_MODEL,
    DIAGNOSTIC_CAMERA_POSES,
    DiagnosticCameraPoseCfg,
)
from ...runtime.xform_spawner import XformSpawnerCfg


def make_diagnostic_camera_cfg(pose_cfg: DiagnosticCameraPoseCfg) -> CameraCfg:
    """按 Test 相机模型创建保留原始分割 ID 的诊断传感器。"""

    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{pose_cfg.prim_name}",
        update_period=0.0,
        height=DIAGNOSTIC_CAMERA_MODEL.height,
        width=DIAGNOSTIC_CAMERA_MODEL.width,
        data_types=list(DIAGNOSTIC_CAMERA_DATA_TYPES),
        colorize_semantic_segmentation=False,
        colorize_instance_id_segmentation=False,
        colorize_instance_segmentation=False,
        offset=CameraCfg.OffsetCfg(
            pos=pose_cfg.position_e,
            rot=pose_cfg.rotation_wxyz,
            convention=pose_cfg.pose_convention,
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=DIAGNOSTIC_CAMERA_MODEL.focal_length_mm,
            focus_distance=DIAGNOSTIC_CAMERA_MODEL.focus_distance_mm,
            horizontal_aperture=DIAGNOSTIC_CAMERA_MODEL.horizontal_aperture_mm,
            clipping_range=DIAGNOSTIC_CAMERA_MODEL.clipping_range_m,
        ),
    )


@configclass
class TcpDockingVisionSceneBaseCfg(InteractiveSceneCfg):
    """只加载语义审计所需资产，刻意不加载机器人 articulation 或接触传感器。"""

    station = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/station", spawn=XformSpawnerCfg())
    station_static = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/station/static", spawn=XformSpawnerCfg())
    station_interactive = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/station/interactive", spawn=XformSpawnerCfg())
    workstation = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/station/static/workstation",
        spawn=sim_utils.UsdFileCfg(usd_path=str(asset_path(asset_by_key("workstation")))),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=WORKSTATION_POSE_E.position,
            rot=WORKSTATION_POSE_E.rotation_wxyz,
        ),
    )
    sample_bottle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/station/interactive/ws_interactive_reagent_01_sample_bottle",
        spawn=sim_utils.UsdFileCfg(usd_path=str(asset_path(asset_by_key("sample_bottle")))),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=TARGET_STATES[0].position_e,
            rot=TARGET_STATES[0].rotation_wxyz,
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )



@configclass
class TcpDockingVisionSceneCfg0(TcpDockingVisionSceneBaseCfg):
    camera_cfg = make_diagnostic_camera_cfg(DIAGNOSTIC_CAMERA_POSES[0])


@configclass
class TcpDockingVisionSceneCfg1(TcpDockingVisionSceneBaseCfg):
    camera_cfg_2 = make_diagnostic_camera_cfg(DIAGNOSTIC_CAMERA_POSES[1])


@configclass
class TcpDockingVisionSceneCfg2(TcpDockingVisionSceneBaseCfg):
    camera_cfg_3 = make_diagnostic_camera_cfg(DIAGNOSTIC_CAMERA_POSES[2])


DIAGNOSTIC_VISION_SCENE_CFGS = (
    TcpDockingVisionSceneCfg0,
    TcpDockingVisionSceneCfg1,
    TcpDockingVisionSceneCfg2,
)
