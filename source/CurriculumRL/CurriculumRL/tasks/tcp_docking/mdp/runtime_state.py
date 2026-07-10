"""从 Isaac 场景张量生成阶段 D 的单步任务状态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ....configs.assets import ROBOT_PRIM_CONTRACT, SCENE_ENTITY_AUBO, SCENE_ENTITY_TARGET
from ....configs.curriculum import CURRICULUM_CFG
from ....configs.rewards import (
    ORIENTATION_QUALITY_SCALE_RAD,
    PROXIMITY_LENGTH_SCALE_M,
    SPEED_QUALITY_SCALE_M_S,
)
from ....configs.task import (
    FLANGE_TO_TOOL_ROTATION_F,
    ILLEGAL_CONTACT_FORCE_N,
    MAX_TARGET_DISPLACEMENT_M,
    MAX_TARGET_LINEAR_SPEED_M_S,
    PARKING_DWELL_POLICY_STEPS,
    PARKING_ENTER_DISTANCE_M,
    PARKING_EXIT_DISTANCE_M,
    PARKING_MAX_ORIENTATION_ERROR_RAD,
    PARKING_MAX_TCP_SPEED_M_S,
    TARGET_DOCKING_AXIS,
    TARGET_STATES,
    TARGET_TO_TOOL_ROTATION_T,
    TCP_WORKSPACE_B,
    TOOL_FORWARD_AXIS,
)
from ....logic.curriculum_state import CurriculumController, EpisodeResult
from ....logic.orientation import tool_axis_alignment
from ....logic.parking_state import ParkingState, initial_parking_state, update_parking_state
from ....logic.reset_state import ResetCache, commit_target_readback, create_reset_cache, prepare_target_reset
from ....logic.rewards import RewardComponents, RewardState, create_reward_state, update_reward_state
from ....logic.tcp_kinematics import quaternion_conjugate, quaternion_multiply, rotate_vector, world_to_root_position
from ....logic.terminations import illegal_contact, outside_workspace, target_disturbed


@dataclass(frozen=True)
class StepMetrics:
    tcp_position_b: torch.Tensor
    position_error_b: torch.Tensor
    orientation_error_vector_b: torch.Tensor
    tcp_velocity_b: torch.Tensor
    distance_m: torch.Tensor
    orientation_error_rad: torch.Tensor
    tcp_speed_m_s: torch.Tensor
    axis_alignment: torch.Tensor
    parking_inside: torch.Tensor
    success: torch.Tensor
    outside_workspace: torch.Tensor
    illegal_contact: torch.Tensor
    target_disturbed: torch.Tensor

    @property
    def safety_failure(self) -> torch.Tensor:
        return self.outside_workspace | self.illegal_contact | self.target_disturbed


@dataclass
class TcpDockingRuntimeState:
    reset_cache: ResetCache
    parking_state: ParkingState
    reward_state: RewardState
    last_step: torch.Tensor
    curriculum: CurriculumController
    episode_level: torch.Tensor
    episode_active: torch.Tensor
    completed_position_error_m: torch.Tensor
    completed_orientation_error_rad: torch.Tensor
    completed_tcp_speed_m_s: torch.Tensor
    completed_success: torch.Tensor
    completed_safety_failure: torch.Tensor
    completed_timeout: torch.Tensor
    metrics: StepMetrics | None = None
    rewards: RewardComponents | None = None


def get_runtime(env: Any) -> TcpDockingRuntimeState:
    state = getattr(env, "_tcp_docking_runtime", None)
    if state is not None:
        return state
    cache = create_reset_cache(env.num_envs, device=env.device)
    state_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    _prepare_cache(env, cache, torch.arange(env.num_envs, device=env.device), state_indices)
    state = TcpDockingRuntimeState(
        reset_cache=cache,
        parking_state=initial_parking_state(env.num_envs, device=env.device),
        reward_state=create_reward_state(env.num_envs, device=env.device),
        last_step=torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
        curriculum=CurriculumController(CURRICULUM_CFG, level=int(getattr(env.cfg, "curriculum_initial_level", 0))),
        episode_level=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        episode_active=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        completed_position_error_m=torch.full((env.num_envs,), torch.nan, device=env.device),
        completed_orientation_error_rad=torch.full((env.num_envs,), torch.nan, device=env.device),
        completed_tcp_speed_m_s=torch.full((env.num_envs,), torch.nan, device=env.device),
        completed_success=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        completed_safety_failure=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        completed_timeout=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    env._tcp_docking_runtime = state
    return state


def reset_runtime(env: Any, env_ids: torch.Tensor, state_indices: torch.Tensor) -> None:
    state = get_runtime(env)
    _prepare_cache(env, state.reset_cache, env_ids, state_indices, write_target=True)
    state.parking_state.inside[env_ids] = False
    state.parking_state.dwell_steps[env_ids] = 0
    state.parking_state.last_control_step[env_ids] = -1
    state.parking_state.success[env_ids] = False
    state.last_step[env_ids] = -1
    from ....logic.rewards import reset_reward_state

    reset_reward_state(state.reward_state, env_ids)
    state.metrics = None


def reset_curriculum_runtime(env: Any, env_ids: torch.Tensor) -> None:
    """结算旧 episode 后切换一次全局课程，并为 reset 环境采样下一目标。"""

    state = get_runtime(env)
    _record_completed_episodes(env, state, env_ids)
    if getattr(env.cfg, "curriculum_enabled", True):
        level = state.curriculum.level
        probabilities = torch.tensor(
            CURRICULUM_CFG.levels[level].target_state_probabilities,
            dtype=env.scene.env_origins.dtype,
            device=env.device,
        )
        state_indices = torch.multinomial(probabilities, env_ids.numel(), replacement=True)
        state.episode_level[env_ids] = state.curriculum.level
    else:
        forced_index = getattr(env.cfg, "evaluation_target_state_index", None)
        if forced_index is None or not 0 <= forced_index < len(TARGET_STATES):
            raise ValueError("固定评估必须指定有效的 evaluation_target_state_index")
        evaluation_level = getattr(env.cfg, "evaluation_curriculum_level", len(CURRICULUM_CFG.levels) - 1)
        if not 0 <= evaluation_level < len(CURRICULUM_CFG.levels):
            raise ValueError("固定评估必须指定有效的 evaluation_curriculum_level")
        state_indices = torch.full_like(env_ids, forced_index)
        state.episode_level[env_ids] = evaluation_level
    state.episode_active[env_ids] = True
    reset_runtime(env, env_ids, state_indices)


def curriculum_level_normalized(env: Any) -> torch.Tensor:
    """返回每个 episode 的归一化等级，保持阶段 D 的一维观测形状。"""

    state = get_runtime(env)
    maximum = max(len(CURRICULUM_CFG.levels) - 1, 1)
    return (state.episode_level.to(dtype=env.scene.env_origins.dtype) / maximum).unsqueeze(-1)


def auxiliary_reward_scales(env: Any, component_name: str) -> torch.Tensor:
    """按 episode 归属等级返回逐环境辅助奖励缩放。"""

    if component_name in {"final_success", "safety_failure"}:
        return torch.ones(env.num_envs, dtype=env.scene.env_origins.dtype, device=env.device)
    if component_name not in CURRICULUM_CFG.levels[0].auxiliary_rewards.__dataclass_fields__:
        raise ValueError(f"未知辅助奖励分量：{component_name}")
    state = get_runtime(env)
    values = torch.tensor(
        [getattr(level.auxiliary_rewards, component_name) for level in CURRICULUM_CFG.levels],
        dtype=env.scene.env_origins.dtype,
        device=env.device,
    )
    return values[state.episode_level]


def export_curriculum_state(env: Any) -> dict[str, object]:
    return get_runtime(env).curriculum.snapshot()


def restore_curriculum_state(env: Any, snapshot: dict[str, object]) -> None:
    get_runtime(env).curriculum = CurriculumController.from_snapshot(CURRICULUM_CFG, snapshot)


def _record_completed_episodes(env: Any, state: TcpDockingRuntimeState, env_ids: torch.Tensor) -> None:
    """保存终局指标，并严格让安全失败覆盖同一步停车成功。"""

    active_ids = env_ids[state.episode_active[env_ids]]
    if active_ids.numel() == 0 or state.metrics is None:
        return
    metrics = state.metrics
    safety = metrics.safety_failure[active_ids]
    success = metrics.success[active_ids] & ~safety
    state.completed_position_error_m[active_ids] = metrics.distance_m[active_ids]
    state.completed_orientation_error_rad[active_ids] = metrics.orientation_error_rad[active_ids]
    state.completed_tcp_speed_m_s[active_ids] = metrics.tcp_speed_m_s[active_ids]
    state.completed_success[active_ids] = success
    state.completed_safety_failure[active_ids] = safety
    state.completed_timeout[active_ids] = env.episode_length_buf[active_ids] >= env.max_episode_length - 1
    if getattr(env.cfg, "curriculum_enabled", True):
        state.curriculum.submit_batch(
            EpisodeResult(
                level=int(state.episode_level[env_id].item()),
                success=bool(success[index].item()),
                safety_failure=bool(safety[index].item()),
            )
            for index, env_id in enumerate(active_ids)
        )


def compute_step(env: Any) -> tuple[StepMetrics, RewardComponents]:
    state = get_runtime(env)
    step = env.episode_length_buf
    if state.metrics is not None and state.rewards is not None and torch.equal(state.last_step, step):
        return state.metrics, state.rewards

    action_term = env.action_manager.get_term("tcp_delta")
    controller = action_term.controller
    robot = env.scene[SCENE_ENTITY_AUBO]
    target = env.scene[SCENE_ENTITY_TARGET]
    tcp_position_b, tcp_quaternion_b, flange_quaternion_b = controller.current_tcp_pose_b()
    root_quaternion_w = robot.data.root_quat_w
    desired_tool_w = quaternion_multiply(
        state.reset_cache.target_quaternion_wxyz,
        torch.tensor(TARGET_TO_TOOL_ROTATION_T, dtype=tcp_position_b.dtype, device=env.device).expand(env.num_envs, -1),
    )
    desired_tool_b = quaternion_multiply(quaternion_conjugate(root_quaternion_w), desired_tool_w)
    orientation_error_vector_b = _rotation_error_vector(tcp_quaternion_b, desired_tool_b)
    orientation_error_rad = torch.linalg.vector_norm(orientation_error_vector_b, dim=-1)
    preposition_b = world_to_root_position(state.reset_cache.preposition_w, robot.data.root_pos_w, root_quaternion_w)
    position_error_b = preposition_b - tcp_position_b
    distance_m = torch.linalg.vector_norm(position_error_b, dim=-1)

    flange_linear_w = robot.data.body_lin_vel_w[:, controller.flange_body_id]
    flange_angular_w = robot.data.body_ang_vel_w[:, controller.flange_body_id]
    offset_w = rotate_vector(
        robot.data.body_quat_w[:, controller.flange_body_id], controller.flange_to_tcp_translation_f
    )
    tcp_linear_w = flange_linear_w + torch.linalg.cross(flange_angular_w, offset_w)
    root_inverse = quaternion_conjugate(root_quaternion_w)
    tcp_velocity_b = torch.cat(
        (rotate_vector(root_inverse, tcp_linear_w), rotate_vector(root_inverse, flange_angular_w)), dim=-1
    )
    tcp_speed_m_s = torch.linalg.vector_norm(tcp_velocity_b[:, :3], dim=-1)
    _, axis_alignment = tool_axis_alignment(
        robot.data.body_quat_w[:, controller.flange_body_id],
        state.reset_cache.target_quaternion_wxyz,
        torch.tensor(FLANGE_TO_TOOL_ROTATION_F, dtype=tcp_position_b.dtype, device=env.device),
        torch.tensor(TOOL_FORWARD_AXIS, dtype=tcp_position_b.dtype, device=env.device),
        torch.tensor(TARGET_DOCKING_AXIS, dtype=tcp_position_b.dtype, device=env.device),
    )
    state.parking_state = update_parking_state(
        state.parking_state,
        distance_m,
        tcp_speed_m_s,
        orientation_error_rad,
        step,
        enter_distance_m=PARKING_ENTER_DISTANCE_M,
        exit_distance_m=PARKING_EXIT_DISTANCE_M,
        max_tcp_speed_m_s=PARKING_MAX_TCP_SPEED_M_S,
        max_orientation_error_rad=PARKING_MAX_ORIENTATION_ERROR_RAD,
        required_dwell_steps=PARKING_DWELL_POLICY_STEPS,
    )
    contact = env.scene["robot_contact"]
    contact_failure = illegal_contact(
        contact.data.net_forces_w,
        tuple(contact.body_names),
        ROBOT_PRIM_CONTRACT.ignored_contact_bodies,
        ILLEGAL_CONTACT_FORCE_N,
    )
    disturbed = target_disturbed(
        target.data.root_pos_w,
        state.reset_cache.target_baseline_position_w,
        target.data.root_lin_vel_w,
        MAX_TARGET_DISPLACEMENT_M,
        MAX_TARGET_LINEAR_SPEED_M_S,
    )
    workspace = torch.tensor(TCP_WORKSPACE_B, dtype=tcp_position_b.dtype, device=env.device)
    metrics = StepMetrics(
        tcp_position_b=tcp_position_b,
        position_error_b=position_error_b,
        orientation_error_vector_b=orientation_error_vector_b,
        tcp_velocity_b=tcp_velocity_b,
        distance_m=distance_m,
        orientation_error_rad=orientation_error_rad,
        tcp_speed_m_s=tcp_speed_m_s,
        axis_alignment=axis_alignment,
        parking_inside=state.parking_state.inside,
        success=state.parking_state.success,
        outside_workspace=outside_workspace(tcp_position_b, workspace),
        illegal_contact=contact_failure,
        target_disturbed=disturbed,
    )
    update_mask = state.last_step != step
    candidate_rewards = update_reward_state(
        state.reward_state,
        distance_m,
        orientation_error_rad,
        tcp_speed_m_s,
        axis_alignment,
        metrics.parking_inside,
        metrics.success,
        metrics.safety_failure,
        proximity_length_scale_m=PROXIMITY_LENGTH_SCALE_M,
        orientation_quality_scale_rad=ORIENTATION_QUALITY_SCALE_RAD,
        speed_quality_scale_m_s=SPEED_QUALITY_SCALE_M_S,
        update_mask=update_mask,
    )
    if state.rewards is None:
        rewards = candidate_rewards
    else:
        rewards = RewardComponents(
            **{
                name: torch.where(update_mask, value, getattr(state.rewards, name))
                for name, value in vars(candidate_rewards).items()
            }
        )
    state.last_step[update_mask] = step[update_mask]
    state.metrics = metrics
    state.rewards = rewards
    return metrics, rewards


def _prepare_cache(
    env: Any,
    cache: ResetCache,
    env_ids: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    write_target: bool = False,
) -> None:
    dtype = env.scene.env_origins.dtype
    device = env.device
    target_pose_w, target_velocity_w = prepare_target_reset(
        cache,
        env_ids,
        state_indices,
        env.scene.env_origins,
        torch.tensor([state.position_e for state in TARGET_STATES], dtype=dtype, device=device),
        torch.tensor([state.rotation_wxyz for state in TARGET_STATES], dtype=dtype, device=device),
        torch.tensor([state.preposition_e for state in TARGET_STATES], dtype=dtype, device=device),
        torch.tensor(TARGET_TO_TOOL_ROTATION_T, dtype=dtype, device=device),
        torch.tensor(FLANGE_TO_TOOL_ROTATION_F, dtype=dtype, device=device),
    )
    target = env.scene[SCENE_ENTITY_TARGET]
    if write_target:
        target.write_root_pose_to_sim(target_pose_w, env_ids=env_ids)
        target.write_root_velocity_to_sim(target_velocity_w, env_ids=env_ids)
    actual_position = target_pose_w[:, :3] if write_target else target.data.root_pos_w[env_ids]
    actual_quaternion = target_pose_w[:, 3:7] if write_target else target.data.root_quat_w[env_ids]
    commit_target_readback(
        cache,
        env_ids,
        actual_position,
        actual_quaternion,
        torch.tensor(TARGET_TO_TOOL_ROTATION_T, dtype=dtype, device=device),
        torch.tensor(FLANGE_TO_TOOL_ROTATION_F, dtype=dtype, device=device),
    )


def _rotation_error_vector(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    from ....logic.differential_ik import quaternion_to_rotation_vector

    error = quaternion_multiply(target, quaternion_conjugate(current))
    error = torch.where(error[:, :1] < 0.0, -error, error)
    return quaternion_to_rotation_vector(error)
