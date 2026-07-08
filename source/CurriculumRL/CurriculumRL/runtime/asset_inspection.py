"""阶段 A 的 USD 资产与 articulation 静态检查。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..configs.assets import ASSETS, ROBOT_PRIM_CONTRACT, AssetSpec, asset_path, resolve_asset_root


@dataclass(frozen=True)
class AssetInspectionResult:
    """单个资产的可序列化检查结果。"""

    key: str
    path: str
    required: bool
    exists: bool
    stage_opened: bool = False
    default_prim: str | None = None
    dependencies: tuple[str, ...] = ()
    unresolved_dependencies: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """必需资产存在、可打开且不存在契约错误。"""

        if not self.required and not self.exists:
            return True
        return self.exists and self.stage_opened and not self.unresolved_dependencies and not self.errors

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典。"""

        data = asdict(self)
        data["passed"] = self.passed
        return data


def check_asset_files(asset_root: str | Path | None = None) -> list[tuple[AssetSpec, Path, bool]]:
    """在启动 Kit 前快速检查资产清单。"""

    root = resolve_asset_root(asset_root)
    return [(spec, asset_path(spec, root), asset_path(spec, root).is_file()) for spec in ASSETS]


def inspect_asset_stages(asset_root: str | Path | None = None) -> list[AssetInspectionResult]:
    """使用已启动的 Kit/pxr 递归检查 USD 依赖和机器人 prim 契约。"""

    from pxr import Usd, UsdUtils

    results: list[AssetInspectionResult] = []
    for spec, path, exists in check_asset_files(asset_root):
        if not spec.required:
            continue
        errors: list[str] = []
        dependencies: tuple[str, ...] = ()
        unresolved: tuple[str, ...] = ()
        default_prim: str | None = None
        stage_opened = False

        if exists:
            stage = Usd.Stage.Open(str(path))
            stage_opened = bool(stage)
            if stage:
                default = stage.GetDefaultPrim()
                default_prim = default.GetName() if default else None
                if default_prim is None:
                    errors.append("USD 未声明 default prim")

                _, dependency_paths, unresolved_paths = UsdUtils.ComputeAllDependencies(str(path))
                dependencies = tuple(sorted(str(item) for item in dependency_paths))
                unresolved = tuple(sorted(str(item) for item in unresolved_paths))

                if spec.key == "aubo_with_gripper":
                    errors.extend(_inspect_robot_contract(stage))
                if spec.key == "workstation" and not _has_schema(stage, "PhysicsCollisionAPI"):
                    errors.append("工作站 USD 未发现 PhysicsCollisionAPI")
            else:
                errors.append("Usd.Stage.Open 返回空 stage")
        elif spec.required:
            errors.append(f"缺少必需资产：{path}")

        results.append(
            AssetInspectionResult(
                key=spec.key,
                path=str(path),
                required=spec.required,
                exists=exists,
                stage_opened=stage_opened,
                default_prim=default_prim,
                dependencies=dependencies,
                unresolved_dependencies=unresolved,
                errors=tuple(errors),
            )
        )
    return results


def _inspect_robot_contract(stage: Any) -> list[str]:
    contract = ROBOT_PRIM_CONTRACT
    errors: list[str] = []
    default = stage.GetDefaultPrim()
    if not default or default.GetName() != contract.usd_default_prim:
        actual = default.GetName() if default else None
        errors.append(f"机器人 default prim 应为 {contract.usd_default_prim!r}，实际为 {actual!r}")
        return errors

    articulation = stage.GetPrimAtPath(default.GetPath().AppendChild(contract.articulation_prim))
    if not articulation:
        errors.append(f"缺少 articulation prim：{contract.articulation_prim}")
        return errors
    if "PhysicsArticulationRootAPI" not in articulation.GetAppliedSchemas():
        errors.append(f"prim {articulation.GetPath()} 未应用 PhysicsArticulationRootAPI")

    joint_names = [
        prim.GetName()
        for prim in stage.Traverse()
        if prim.GetTypeName() in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}
    ]
    for name in (*contract.arm_joints, *contract.gripper_joints):
        count = joint_names.count(name)
        if count != 1:
            errors.append(f"关节 {name!r} 应唯一匹配，实际匹配 {count} 次")

    overlap = set(contract.arm_joints).intersection(contract.gripper_joints)
    if overlap:
        errors.append(f"夹爪关节混入机械臂关节契约：{sorted(overlap)}")

    flange_bodies = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() == contract.flange_body and "PhysicsRigidBodyAPI" in prim.GetAppliedSchemas()
    ]
    if len(flange_bodies) != 1:
        errors.append(f"刚体 {contract.flange_body!r} 应唯一匹配，实际匹配 {len(flange_bodies)} 次")
    if not _has_schema(stage, "PhysicsCollisionAPI"):
        errors.append("机器人 USD 未发现 PhysicsCollisionAPI")
    return errors


def _has_schema(stage: Any, schema_name: str) -> bool:
    return any(schema_name in prim.GetAppliedSchemas() for prim in stage.Traverse())
