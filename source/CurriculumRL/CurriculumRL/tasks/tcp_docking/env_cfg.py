"""阶段 E：课程目标采样 TCP 停靠 Manager-Based 环境。"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ...configs.assets import SCENE_ENTITY_AUBO
from ...configs.rewards import (
    BEST_PROGRESS_WEIGHT,
    DISTANCE_PROGRESS_WEIGHT,
    FINAL_SUCCESS_WEIGHT,
    FIRST_ENTRY_WEIGHT,
    INNER_DOCKING_QUALITY_WEIGHT,
    LOW_SPEED_PARKING_WEIGHT,
    PROXIMITY_WEIGHT,
    SAFETY_FAILURE_WEIGHT,
    TOOL_AXIS_PROGRESS_WEIGHT,
)
from ...configs.training import EPISODE_LENGTH_S, POLICY_DECIMATION, SIMULATION_DT_S
from . import mdp
from .actions import DifferentialIKActionTermCfg
from .dynamic_scene_cfg import TcpDockingDynamicSceneCfg


@configclass
class ActionsCfg:
    tcp_delta = DifferentialIKActionTermCfg(asset_name=SCENE_ENTITY_AUBO)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_position = ObsTerm(func=mdp.arm_joint_position)
        joint_velocity = ObsTerm(func=mdp.arm_joint_velocity)
        tcp_position_error = ObsTerm(func=mdp.tcp_position_error)
        tcp_orientation_error = ObsTerm(func=mdp.tcp_orientation_error)
        tcp_velocity = ObsTerm(func=mdp.tcp_velocity)
        target_state = ObsTerm(func=mdp.target_state_one_hot)
        curriculum_level = ObsTerm(func=mdp.curriculum_level)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_curriculum_target = EventTerm(func=mdp.reset_curriculum_target, mode="reset")


@configclass
class RewardsCfg:
    distance_progress = RewTerm(func=mdp.distance_progress, weight=DISTANCE_PROGRESS_WEIGHT)
    best_progress = RewTerm(func=mdp.best_progress, weight=BEST_PROGRESS_WEIGHT)
    proximity = RewTerm(func=mdp.proximity, weight=PROXIMITY_WEIGHT)
    inner_docking_quality = RewTerm(func=mdp.inner_docking_quality, weight=INNER_DOCKING_QUALITY_WEIGHT)
    low_speed_parking = RewTerm(func=mdp.low_speed_parking, weight=LOW_SPEED_PARKING_WEIGHT)
    first_entry = RewTerm(func=mdp.first_entry, weight=FIRST_ENTRY_WEIGHT)
    tool_axis_progress = RewTerm(func=mdp.tool_axis_progress, weight=TOOL_AXIS_PROGRESS_WEIGHT)
    final_success = RewTerm(func=mdp.final_success, weight=FINAL_SUCCESS_WEIGHT)
    safety_failure = RewTerm(func=mdp.safety_failure, weight=SAFETY_FAILURE_WEIGHT)


@configclass
class TerminationsCfg:
    parking_success = DoneTerm(func=mdp.parking_success)
    outside_workspace = DoneTerm(func=mdp.outside_tcp_workspace)
    illegal_contact = DoneTerm(func=mdp.illegal_robot_contact)
    target_disturbed = DoneTerm(func=mdp.target_was_disturbed)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class TcpDockingEnvCfg(ManagerBasedRLEnvCfg):
    scene: TcpDockingDynamicSceneCfg = TcpDockingDynamicSceneCfg(num_envs=64, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum_enabled: bool = True
    curriculum_initial_level: int = 0
    evaluation_target_state_index: int | None = None
    evaluation_curriculum_level: int = 4

    def __post_init__(self) -> None:
        self.decimation = POLICY_DECIMATION
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = SIMULATION_DT_S
        self.sim.render_interval = POLICY_DECIMATION
        self.viewer.eye = (3.2, 2.5, 2.2)
        self.viewer.lookat = (1.2, 0.0, 0.9)
