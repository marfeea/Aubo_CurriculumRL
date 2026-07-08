"""阶段 C 六维增量位姿、DLS 和安全投影纯逻辑测试。"""

import math
import sys
from pathlib import Path

import pytest
import torch

PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "source" / "CurriculumRL"
sys.path.insert(0, str(PACKAGE_SOURCE))

from CurriculumRL.logic.differential_ik import (  # noqa: E402
    apply_tcp_increment,
    damped_least_squares,
    pose_error,
    project_joint_target,
    quaternion_to_rotation_vector,
    rotation_vector_to_quaternion,
    scale_policy_action,
    tcp_jacobian_from_flange,
)

DTYPE = torch.float64


def tensor(value: object) -> torch.Tensor:
    return torch.tensor(value, dtype=DTYPE)


def test_policy_action_is_clipped_and_scaled_per_component() -> None:
    processed = scale_policy_action(
        tensor([[2.0, -2.0, 0.5, 2.0, -2.0, 0.5]]),
        action_clip=1.0,
        position_scale_m=0.02,
        rotation_scale_rad=0.1,
    )
    torch.testing.assert_close(processed, tensor([[0.02, -0.02, 0.01, 0.1, -0.1, 0.05]]))
    with pytest.raises(ValueError, match="NaN/Inf"):
        scale_policy_action(
            tensor([[math.nan, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            action_clip=1.0,
            position_scale_m=0.02,
            rotation_scale_rad=0.1,
        )


def test_rotation_vector_round_trip_handles_zero_and_pi() -> None:
    rotation_vectors = tensor([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [math.pi, 0.0, 0.0]])
    recovered = quaternion_to_rotation_vector(rotation_vector_to_quaternion(rotation_vectors))
    torch.testing.assert_close(recovered, rotation_vectors, atol=1e-7, rtol=1e-7)


def test_tcp_increment_and_pose_error_use_root_frame_rotation_vector() -> None:
    position = tensor([[1.0, 2.0, 3.0]])
    quaternion = tensor([[1.0, 0.0, 0.0, 0.0]])
    action = tensor([[0.01, -0.02, 0.03, 0.0, 0.0, 0.1]])
    target_position, target_quaternion = apply_tcp_increment(position, quaternion, action)
    error = pose_error(
        position,
        quaternion,
        target_position,
        target_quaternion,
        position_gain=1.0,
        rotation_gain=1.0,
    )
    torch.testing.assert_close(error, action, atol=1e-7, rtol=1e-7)


def test_tcp_jacobian_contains_offset_tangential_term() -> None:
    flange_jacobian = torch.zeros((1, 6, 1), dtype=DTYPE)
    flange_jacobian[0, 5, 0] = 1.0  # 绕 B 系 z 轴的单位角速度。
    tcp_jacobian = tcp_jacobian_from_flange(
        flange_jacobian,
        tensor([[1.0, 0.0, 0.0, 0.0]]),
        tensor([1.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(tcp_jacobian[0, :, 0], tensor([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))


def test_dls_solves_full_rank_and_raises_damping_near_singularity() -> None:
    identity = torch.eye(6, dtype=DTYPE).unsqueeze(0)
    task_error = tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]])
    result = damped_least_squares(
        identity,
        task_error,
        damping=0.1,
        singular_value_threshold=0.05,
        singular_damping=0.2,
    )
    torch.testing.assert_close(result.joint_delta, task_error / 1.01)
    assert result.singular.tolist() == [False]
    singular_jacobian = identity.clone()
    singular_jacobian[:, -1, -1] = 0.0
    protected = damped_least_squares(
        singular_jacobian,
        task_error,
        damping=0.1,
        singular_value_threshold=0.05,
        singular_damping=0.2,
    )
    assert protected.singular.tolist() == [True]
    torch.testing.assert_close(protected.damping, tensor([0.2]))
    assert torch.isfinite(protected.joint_delta).all()


def test_joint_target_projection_enforces_step_velocity_and_position_limits() -> None:
    current = tensor([[0.0, 0.0, 0.99]])
    requested = tensor([[1.0, 0.5, 0.5]])
    position_limits = tensor([[[-2.0, 2.0], [-2.0, 2.0], [-1.0, 1.0]]])
    velocity_limits = tensor([[10.0, 0.1, 10.0]])
    result = project_joint_target(
        current,
        requested,
        position_limits,
        velocity_limits,
        physics_dt_s=0.1,
        max_joint_delta_rad=0.2,
        max_joint_velocity_rad_s=1.0,
        joint_position_margin_rad=0.01,
    )
    torch.testing.assert_close(result.joint_target, tensor([[0.1, 0.01, 0.99]]))
    assert result.delta_limited.tolist() == [True]
    assert result.velocity_limited.tolist() == [True]
    assert result.position_limited.tolist() == [True]


def test_dls_rejects_wrong_dimensions_before_solver() -> None:
    with pytest.raises(ValueError, match="env,6,joint"):
        damped_least_squares(
            torch.zeros((2, 5, 6), dtype=DTYPE),
            torch.zeros((2, 6), dtype=DTYPE),
            damping=0.1,
            singular_value_threshold=0.05,
            singular_damping=0.2,
        )
