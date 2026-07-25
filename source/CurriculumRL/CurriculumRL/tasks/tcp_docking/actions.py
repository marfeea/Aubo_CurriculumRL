"""AUBO 六维 TCP 增量到安全关节位置目标的 Isaac 适配链。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ...configs.assets import ROBOT_PRIM_CONTRACT
from ...configs.differential_ik import (
    ACTION_CLIP,
    DLS_DAMPING,
    JOINT_POSITION_MARGIN_RAD,
    MAX_JOINT_DELTA_RAD,
    MAX_JOINT_VELOCITY_RAD_S,
    POSITION_ERROR_GAIN,
    POSITION_INCREMENT_SCALE_M,
    ROTATION_ERROR_GAIN,
    ROTATION_INCREMENT_SCALE_RAD,
    SINGULAR_DAMPING,
    SINGULAR_VALUE_THRESHOLD,
)
from ...configs.task import FLANGE_TO_TCP_TRANSLATION_F, FLANGE_TO_TOOL_ROTATION_F
from ...logic.differential_ik import (
    JointProjectionResult,
    apply_tcp_increment,
    damped_least_squares,
    pose_error,
    project_joint_target,
    scale_policy_action,
    tcp_jacobian_from_flange,
)
from ...logic.tcp_kinematics import (
    quaternion_conjugate,
    quaternion_multiply,
    rotate_vector,
    world_to_root_position,
)


@dataclass(frozen=True)
class DifferentialIKDiagnostics:
    raw_action: torch.Tensor
    processed_action: torch.Tensor
    current_tcp_position_b: torch.Tensor
    target_tcp_position_b: torch.Tensor
    task_error: torch.Tensor
    minimum_singular_value: torch.Tensor
    damping: torch.Tensor
    singular: torch.Tensor
    requested_joint_delta: torch.Tensor
    applied_joint_delta: torch.Tensor
    delta_limited: torch.Tensor
    velocity_limited: torch.Tensor
    position_limited: torch.Tensor
    joint_target: torch.Tensor


class DifferentialIKAction:
    """由策略步设置 TCP 增量，并在每个物理步持续跟踪同一目标。"""

    def __init__(self, robot: Any, physics_dt_s: float, *, diagnostics_enabled: bool = False):
        self.robot = robot
        self.physics_dt_s = physics_dt_s
        self.diagnostics_enabled = diagnostics_enabled
        self.joint_ids, joint_names = robot.find_joints(list(ROBOT_PRIM_CONTRACT.arm_joints), preserve_order=True)
        flange_ids, flange_names = robot.find_bodies([ROBOT_PRIM_CONTRACT.flange_body], preserve_order=True)
        if tuple(joint_names) != ROBOT_PRIM_CONTRACT.arm_joints:
            raise RuntimeError(
                f"机械臂关节解析不一致：期望 {ROBOT_PRIM_CONTRACT.arm_joints}，实际 {tuple(joint_names)}"
            )
        if tuple(flange_names) != (ROBOT_PRIM_CONTRACT.flange_body,):
            raise RuntimeError(f"Flange body 必须唯一匹配，实际 {tuple(flange_names)}")
        if len(set(self.joint_ids)) != 6:
            raise RuntimeError(f"机械臂关节索引必须是六个唯一值，实际 {self.joint_ids}")
        self.flange_body_id = flange_ids[0]
        self.jacobian_body_id = self.flange_body_id - 1 if robot.is_fixed_base else self.flange_body_id
        self.jacobian_joint_ids = self.joint_ids if robot.is_fixed_base else [index + 6 for index in self.joint_ids]
        self._validate_jacobian_contract()

        num_envs = robot.data.joint_pos.shape[0]
        device = robot.data.joint_pos.device
        dtype = robot.data.joint_pos.dtype
        self.flange_to_tcp_translation_f = torch.tensor(FLANGE_TO_TCP_TRANSLATION_F, dtype=dtype, device=device).expand(
            num_envs, -1
        )
        self.flange_to_tool_rotation_f = torch.tensor(FLANGE_TO_TOOL_ROTATION_F, dtype=dtype, device=device).expand(
            num_envs, -1
        )
        self.raw_action = torch.zeros((num_envs, 6), dtype=dtype, device=device)
        self.processed_action = torch.zeros_like(self.raw_action)
        self.target_tcp_position_b = torch.zeros((num_envs, 3), dtype=dtype, device=device)
        self.target_tcp_quaternion_b = torch.zeros((num_envs, 4), dtype=dtype, device=device)
        self._diagnostics: DifferentialIKDiagnostics | None = None
        self.reset()

    @property
    def diagnostics(self) -> DifferentialIKDiagnostics | None:
        return self._diagnostics if self.diagnostics_enabled else None

    def _validate_jacobian_contract(self) -> None:
        jacobians = self.robot.root_physx_view.get_jacobians()
        if jacobians.ndim != 4 or jacobians.shape[2] != 6:
            raise RuntimeError(f"Jacobian 必须为 (env, body, 6, dof)，实际 {tuple(jacobians.shape)}")
        if not 0 <= self.jacobian_body_id < jacobians.shape[1]:
            raise RuntimeError(
                f"Flange body_id={self.flange_body_id}、fixed_base={self.robot.is_fixed_base} 得到非法 "
                f"jacobian_body_id={self.jacobian_body_id}，原始形状={tuple(jacobians.shape)}"
            )
        if min(self.jacobian_joint_ids) < 0 or max(self.jacobian_joint_ids) >= jacobians.shape[3]:
            raise RuntimeError(
                f"机械臂 joint_ids={self.joint_ids}、fixed_base={self.robot.is_fixed_base} 得到非法 Jacobian "
                f"列={self.jacobian_joint_ids}，原始形状={tuple(jacobians.shape)}"
            )

    def current_tcp_pose_b(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回当前 TCP 的 B 系位置/姿态及法兰 B 系姿态。"""

        flange_position_w = self.robot.data.body_pos_w[:, self.flange_body_id]
        flange_quaternion_w = self.robot.data.body_quat_w[:, self.flange_body_id]
        root_position_w = self.robot.data.root_pos_w
        root_quaternion_w = self.robot.data.root_quat_w
        offset_w = rotate_vector(flange_quaternion_w, self.flange_to_tcp_translation_f)
        tcp_position_b = world_to_root_position(flange_position_w + offset_w, root_position_w, root_quaternion_w)
        flange_quaternion_b = quaternion_multiply(quaternion_conjugate(root_quaternion_w), flange_quaternion_w)
        tcp_quaternion_b = quaternion_multiply(flange_quaternion_b, self.flange_to_tool_rotation_f)
        return tcp_position_b, tcp_quaternion_b, flange_quaternion_b

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """将选中环境目标设为当前 TCP，保证 reset 后零动作保持。"""

        if env_ids is None:
            env_ids = torch.arange(self.raw_action.shape[0], device=self.raw_action.device)
        position_b, quaternion_b, _ = self.current_tcp_pose_b()
        self.raw_action[env_ids] = 0.0
        self.processed_action[env_ids] = 0.0
        self.target_tcp_position_b[env_ids] = position_b[env_ids]
        self.target_tcp_quaternion_b[env_ids] = quaternion_b[env_ids]

    def process_actions(self, raw_action: torch.Tensor, *, action_mask: torch.Tensor | None = None) -> None:
        """每个策略步调用一次，基于当前 TCP 生成并缓存六维增量目标。"""

        if raw_action.shape != self.raw_action.shape:
            raise ValueError(f"动作形状必须为 {tuple(self.raw_action.shape)}，实际 {tuple(raw_action.shape)}")
        if action_mask is None:
            action_mask = torch.ones_like(raw_action)
        if action_mask.shape != raw_action.shape:
            raise ValueError(f"动作掩码形状必须为 {tuple(raw_action.shape)}，实际 {tuple(action_mask.shape)}")
        self.raw_action.copy_(raw_action)
        self.processed_action.copy_(
            scale_policy_action(
                raw_action * action_mask,
                action_clip=ACTION_CLIP,
                position_scale_m=POSITION_INCREMENT_SCALE_M,
                rotation_scale_rad=ROTATION_INCREMENT_SCALE_RAD,
            )
        )
        current_position_b, current_quaternion_b, _ = self.current_tcp_pose_b()
        target_position_b, target_quaternion_b = apply_tcp_increment(
            current_position_b, current_quaternion_b, self.processed_action
        )
        self.target_tcp_position_b.copy_(target_position_b)
        self.target_tcp_quaternion_b.copy_(target_quaternion_b)

    def _flange_jacobian_b(self) -> torch.Tensor:
        jacobian_w = self.robot.root_physx_view.get_jacobians()[
            :, self.jacobian_body_id, :, self.jacobian_joint_ids
        ].clone()
        root_inverse = quaternion_conjugate(self.robot.data.root_quat_w)
        linear_b = rotate_vector(root_inverse.unsqueeze(1), jacobian_w[:, :3, :].transpose(1, 2)).transpose(1, 2)
        angular_b = rotate_vector(root_inverse.unsqueeze(1), jacobian_w[:, 3:, :].transpose(1, 2)).transpose(1, 2)
        return torch.cat((linear_b, angular_b), dim=1)

    def apply_actions(self) -> torch.Tensor:
        """每个物理步求解并写入六关节位置目标，返回最终目标张量。"""

        current_position_b, current_quaternion_b, flange_quaternion_b = self.current_tcp_pose_b()
        task_error = pose_error(
            current_position_b,
            current_quaternion_b,
            self.target_tcp_position_b,
            self.target_tcp_quaternion_b,
            position_gain=POSITION_ERROR_GAIN,
            rotation_gain=ROTATION_ERROR_GAIN,
        )
        tcp_jacobian_b = tcp_jacobian_from_flange(
            self._flange_jacobian_b(), flange_quaternion_b, self.flange_to_tcp_translation_f
        )
        dls = damped_least_squares(
            tcp_jacobian_b,
            task_error,
            damping=DLS_DAMPING,
            singular_value_threshold=SINGULAR_VALUE_THRESHOLD,
            singular_damping=SINGULAR_DAMPING,
        )
        projection = self._project_joint_target(dls.joint_delta)
        self.robot.set_joint_position_target(projection.joint_target, joint_ids=self.joint_ids)
        if self.diagnostics_enabled:
            self._diagnostics = DifferentialIKDiagnostics(
                raw_action=self.raw_action.clone(),
                processed_action=self.processed_action.clone(),
                current_tcp_position_b=current_position_b.clone(),
                target_tcp_position_b=self.target_tcp_position_b.clone(),
                task_error=task_error.clone(),
                minimum_singular_value=dls.minimum_singular_value.clone(),
                damping=dls.damping.clone(),
                singular=dls.singular.clone(),
                requested_joint_delta=dls.joint_delta.clone(),
                applied_joint_delta=projection.joint_delta.clone(),
                delta_limited=projection.delta_limited.clone(),
                velocity_limited=projection.velocity_limited.clone(),
                position_limited=projection.position_limited.clone(),
                joint_target=projection.joint_target.clone(),
            )
        return projection.joint_target

    def _project_joint_target(self, requested_joint_delta: torch.Tensor) -> JointProjectionResult:
        joint_position = self.robot.data.joint_pos[:, self.joint_ids]
        return project_joint_target(
            joint_position,
            requested_joint_delta,
            self.robot.data.joint_pos_limits[:, self.joint_ids],
            self.robot.data.joint_vel_limits[:, self.joint_ids],
            physics_dt_s=self.physics_dt_s,
            max_joint_delta_rad=MAX_JOINT_DELTA_RAD,
            max_joint_velocity_rad_s=MAX_JOINT_VELOCITY_RAD_S,
            joint_position_margin_rad=JOINT_POSITION_MARGIN_RAD,
        )


class DifferentialIKActionTerm(ActionTerm):
    """将阶段 C 控制器接入 Manager-Based 动作生命周期。"""

    cfg: DifferentialIKActionTermCfg

    def __init__(self, cfg: DifferentialIKActionTermCfg, env: Any):
        super().__init__(cfg, env)
        self.controller = DifferentialIKAction(
            self._asset,
            env.physics_dt,
            diagnostics_enabled=cfg.diagnostics_enabled,
        )

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self.controller.raw_action

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.controller.processed_action

    def process_actions(self, actions: torch.Tensor) -> None:
        # 延迟导入避免 runtime_state 在 compute_step 中反向导入动作项。
        from .mdp.runtime_state import tcp_action_mask

        self.controller.process_actions(actions, action_mask=tcp_action_mask(self._env))

    def apply_actions(self) -> None:
        self.controller.apply_actions()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.controller.reset(env_ids)


@configclass
class DifferentialIKActionTermCfg(ActionTermCfg):
    class_type: type[ActionTerm] = DifferentialIKActionTerm
    diagnostics_enabled: bool = False
