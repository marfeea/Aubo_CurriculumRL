"""阶段 A 单环境静态场景配置，不包含动作、奖励或训练逻辑。"""

from __future__ import annotations

from isaaclab import sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from ...configs.assets import ROBOT_PRIM_CONTRACT, asset_by_key, asset_path
from ...configs.scene import (
    ARM_ACTUATOR,
    ARM_INITIAL_JOINT_POSITIONS,
    AUBO_2_LOCAL_POSITION_STATION,
    AUBO_LOCAL_POSITION_STATION,
    GRIPPER_ACTUATOR,
    GRIPPER_INITIAL_JOINT_POSITIONS,
    WORKSTATION_POSE_E,
)
from ...configs.task import TARGET_STATES
from ...runtime.xform_spawner import XformSpawnerCfg


def _rotate_vector_wxyz(
    quaternion: tuple[float, float, float, float], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    """用 wxyz 四元数旋转三维向量。"""

    w, x, y, z = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def workstation_local_to_env_position(local_position: tuple[float, float, float]) -> tuple[float, float, float]:
    """将工作站局部位置转换为环境局部位置。"""

    rotated = _rotate_vector_wxyz(WORKSTATION_POSE_E.rotation_wxyz, local_position)
    return tuple(origin + offset for origin, offset in zip(WORKSTATION_POSE_E.position, rotated, strict=True))


def _robot_usd_cfg(entity_name: str, local_position: tuple[float, float, float]) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{entity_name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset_path(asset_by_key("aubo_with_gripper"))),
            activate_contact_sensors=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=workstation_local_to_env_position(local_position),
            rot=WORKSTATION_POSE_E.rotation_wxyz,
        ),
    )


def _robot_articulation_cfg(entity_name: str) -> ArticulationCfg:
    """绑定已由父资产加载的 ``AUBO_E5`` articulation 子 prim。"""

    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{entity_name}/{ROBOT_PRIM_CONTRACT.articulation_prim}",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=workstation_local_to_env_position(
                AUBO_LOCAL_POSITION_STATION if entity_name == "AUBObot" else AUBO_2_LOCAL_POSITION_STATION
            ),
            rot=WORKSTATION_POSE_E.rotation_wxyz,
            joint_pos={**ARM_INITIAL_JOINT_POSITIONS, **GRIPPER_INITIAL_JOINT_POSITIONS},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=list(ROBOT_PRIM_CONTRACT.arm_joints), **ARM_ACTUATOR),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=list(ROBOT_PRIM_CONTRACT.gripper_joints), **GRIPPER_ACTUATOR
            ),
        },
    )


@configclass
class TcpDockingStaticSceneCfg(InteractiveSceneCfg):
    """两台 AUBO、样品瓶和最小工作站子集的静态场景。"""

    station = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/station",
        spawn=XformSpawnerCfg(),
    )
    station_static = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/station/static",
        spawn=XformSpawnerCfg(),
    )
    station_interactive = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/station/interactive",
        spawn=XformSpawnerCfg(),
    )
    station_dynamic = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/station/dynamic",
        spawn=XformSpawnerCfg(),
    )
    workstation = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/station/static/workstation",
        spawn=sim_utils.UsdFileCfg(usd_path=str(asset_path(asset_by_key("workstation")))),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=WORKSTATION_POSE_E.position,
            rot=WORKSTATION_POSE_E.rotation_wxyz,
        ),
    )

    # 机器人 USD 的 default prim 是 /Root，articulation 是其 AUBO_E5 子 prim。
    AUBObot_usd = _robot_usd_cfg("AUBObot", AUBO_LOCAL_POSITION_STATION)
    AUBObot = _robot_articulation_cfg("AUBObot")
    AUBObot_2_usd = _robot_usd_cfg("AUBObot_2", AUBO_2_LOCAL_POSITION_STATION)
    AUBObot_2 = _robot_articulation_cfg("AUBObot_2")

    ws_interactive_reagent_01_sample_bottle = AssetBaseCfg(
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
