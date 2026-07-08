"""阶段 A 资产配置的无 Isaac 回归测试。"""

import sys
from pathlib import Path

PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "source" / "CurriculumRL"
sys.path.insert(0, str(PACKAGE_SOURCE))

from CurriculumRL.configs.assets import (  # noqa: E402
    ASSETS,
    ROBOT_PRIM_CONTRACT,
    asset_by_key,
    asset_path,
    resolve_asset_root,
)
from CurriculumRL.configs.task import TARGET_STATES  # noqa: E402


def test_workspace_asset_root_and_required_files_exist() -> None:
    root = resolve_asset_root()
    assert root.name == "Asset"
    assert all(asset_path(spec, root).is_file() for spec in ASSETS if spec.required)


def test_asset_keys_are_unique() -> None:
    keys = [spec.key for spec in ASSETS]
    assert len(keys) == len(set(keys))
    assert asset_by_key("aubo_with_gripper").relative_path == Path("AUBO_E5/AUBO_E5_Withclaw.usd")


def test_arm_and_gripper_joint_contracts_are_disjoint() -> None:
    assert len(ROBOT_PRIM_CONTRACT.arm_joints) == 6
    assert len(ROBOT_PRIM_CONTRACT.gripper_joints) == 2
    assert set(ROBOT_PRIM_CONTRACT.arm_joints).isdisjoint(ROBOT_PRIM_CONTRACT.gripper_joints)


def test_four_target_states_are_unique() -> None:
    assert len(TARGET_STATES) == 4
    assert len({state.name for state in TARGET_STATES}) == 4
