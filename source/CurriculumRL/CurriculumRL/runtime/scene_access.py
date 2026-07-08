"""阶段 A 静态场景的最小 USD 运行时适配。"""

from __future__ import annotations

from typing import Any

from ..configs.assets import ROBOT_PRIM_CONTRACT
from ..configs.scene import (
    ENABLE_SELF_COLLISIONS,
    SOLVER_POSITION_ITERATIONS,
    SOLVER_VELOCITY_ITERATIONS,
)


def apply_robot_articulation_baseline(stage: Any, expected_num_envs: int = 1) -> tuple[str, ...]:
    """对各环境的两台 articulation 应用迁移基线并返回实际路径。"""

    from pxr import PhysxSchema

    suffixes = (
        f"/AUBObot/{ROBOT_PRIM_CONTRACT.articulation_prim}",
        f"/AUBObot_2/{ROBOT_PRIM_CONTRACT.articulation_prim}",
    )
    matched_paths: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.endswith(suffixes):
            continue
        api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
        api.GetEnabledSelfCollisionsAttr().Set(ENABLE_SELF_COLLISIONS)
        api.GetSolverPositionIterationCountAttr().Set(SOLVER_POSITION_ITERATIONS)
        api.GetSolverVelocityIterationCountAttr().Set(SOLVER_VELOCITY_ITERATIONS)
        matched_paths.append(path)

    expected_count = 2 * expected_num_envs
    if len(matched_paths) != expected_count:
        raise RuntimeError(
            f"{expected_num_envs} 个环境应解析 {expected_count} 台 AUBO articulation，实际为 {matched_paths}"
        )
    return tuple(matched_paths)


def validate_contact_reporting(stage: Any, articulation_path: str) -> None:
    """确认机器人 spawn 已对至少一个刚体激活接触报告。"""

    contact_prims = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(f"{articulation_path}/")
        and "PhysxContactReportAPI" in prim.GetAppliedSchemas()
    ]
    if not contact_prims:
        raise RuntimeError(f"{articulation_path}: 未发现 PhysxContactReportAPI")
