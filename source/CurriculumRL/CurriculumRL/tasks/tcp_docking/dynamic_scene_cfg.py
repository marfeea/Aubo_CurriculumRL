"""阶段 B 目标 reset 与接触张量验证场景。"""

from isaaclab import sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ...configs.assets import ROBOT_CONTACT_PRIM_EXPR, asset_by_key, asset_path
from ...configs.curriculum import CURRICULUM_COLLISION_PROFILES
from ...configs.task import TARGET_STATES
from .static_scene_cfg import TcpDockingStaticSceneCfg


def _c1_contact_sensor_cfg(body_name: str) -> ContactSensorCfg:
    """ContactSensor 仅支持一个传感刚体对多个过滤体，故 C1 按刚体拆分。"""

    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/AUBObot/AUBO_E5/{body_name}",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=list(CURRICULUM_COLLISION_PROFILES[0].filter_prim_paths_expr),
    )


@configclass
class TcpDockingDynamicSceneCfg(TcpDockingStaticSceneCfg):
    """在阶段 A 场景上启用目标刚体视图和第一台机器人接触传感器。"""

    ws_interactive_reagent_01_sample_bottle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/station/interactive/ws_interactive_reagent_01_sample_bottle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset_path(asset_by_key("sample_bottle"))),
            # 阶段 D L0 固定目标姿态；阶段 E 才启用可扰动目标分布。
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TARGET_STATES[0].position_e,
            rot=TARGET_STATES[0].rotation_wxyz,
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )
    robot_contact = ContactSensorCfg(
        prim_path=ROBOT_CONTACT_PRIM_EXPR,
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )
    # P3 的阶段 1 C1 仅将机器人与玻璃防护结构的接触作为非法接触。
    robot_contact_c1_base_link = _c1_contact_sensor_cfg("Base_Link")
    robot_contact_c1_link_01 = _c1_contact_sensor_cfg("Link_01")
    robot_contact_c1_link_02 = _c1_contact_sensor_cfg("Link_02")
    robot_contact_c1_link_03 = _c1_contact_sensor_cfg("Link_03")
    robot_contact_c1_link_04 = _c1_contact_sensor_cfg("Link_04")
    robot_contact_c1_link_05 = _c1_contact_sensor_cfg("Link_05")
    robot_contact_c1_flange = _c1_contact_sensor_cfg("Flange")
