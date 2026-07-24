"""三路视觉语义诊断配置的纯逻辑契约测试。"""

import sys
from pathlib import Path

PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "source" / "CurriculumRL"
sys.path.insert(0, str(PACKAGE_SOURCE))

from CurriculumRL.configs.scene import WORKSTATION_POSE_E  # noqa: E402
from CurriculumRL.configs.vision_diagnostics import (  # noqa: E402
    DIAGNOSTIC_CAMERA_DATA_TYPES,
    DIAGNOSTIC_CAMERA_MODEL,
    DIAGNOSTIC_CAMERA_POSES,
)


def test_test_project_camera_poses_are_migrated_in_environment_frame() -> None:
    assert WORKSTATION_POSE_E.position == (1.3, 0.0, 0.0)
    assert [pose.position_e for pose in DIAGNOSTIC_CAMERA_POSES] == [
        (2.7, 0.0, 1.3),
        (1.3, -0.8, 2.0),
        (1.3, 0.0, 2.8),
    ]
    assert all(pose.pose_convention == "opengl" for pose in DIAGNOSTIC_CAMERA_POSES)


def test_diagnostic_camera_views_are_unique_and_traceable() -> None:
    pose_keys = {(pose.position_e, pose.rotation_wxyz) for pose in DIAGNOSTIC_CAMERA_POSES}
    assert len(pose_keys) == 3
    assert len({pose.scene_name for pose in DIAGNOSTIC_CAMERA_POSES}) == 3
    assert len({pose.prim_name for pose in DIAGNOSTIC_CAMERA_POSES}) == 3
    assert DIAGNOSTIC_CAMERA_POSES[0].source_reference.startswith("Test ")
    assert DIAGNOSTIC_CAMERA_POSES[1].source_reference.startswith("Test ")
    assert DIAGNOSTIC_CAMERA_POSES[2].source_reference == "P0 external top-down diagnostic view"


def test_diagnostic_camera_model_supports_semantic_audit() -> None:
    assert (DIAGNOSTIC_CAMERA_MODEL.width, DIAGNOSTIC_CAMERA_MODEL.height) == (640, 480)
    assert {
        "rgb",
        "distance_to_image_plane",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    }.issubset(DIAGNOSTIC_CAMERA_DATA_TYPES)
