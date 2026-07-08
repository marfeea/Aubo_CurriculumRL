"""为场景语义分组创建不带几何和物理属性的纯 Xform prim。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pxr import Usd

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from typing import Any


@clone
def spawn_xform(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """创建纯 Xform prim，供 `/station` 语义层级使用。"""

    del cfg, kwargs
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"prim 已存在：{prim_path}")
    return create_prim(
        prim_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
        stage=stage,
    )


@configclass
class XformSpawnerCfg(SpawnerCfg):
    """纯 Xform prim 的 Isaac Lab spawner 配置。"""

    func: Callable = spawn_xform
