"""阶段 B 纯逻辑、部分 reset 和安全判定回归测试。"""

import math
import sys
from pathlib import Path

import pytest
import torch

PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "source" / "CurriculumRL"
sys.path.insert(0, str(PACKAGE_SOURCE))

from CurriculumRL.configs.task import (  # noqa: E402
    FLANGE_TO_TCP_TRANSLATION_F,
    FLANGE_TO_TOOL_ROTATION_F,
    PARKING_DWELL_POLICY_STEPS,
    PARKING_ENTER_DISTANCE_M,
    PARKING_EXIT_DISTANCE_M,
    PARKING_MAX_ORIENTATION_ERROR_RAD,
    PARKING_MAX_TCP_SPEED_M_S,
    TARGET_DOCKING_AXIS,
    TARGET_STATES,
    TARGET_TO_TOOL_ROTATION_T,
    TOOL_FORWARD_AXIS,
)
from CurriculumRL.logic.curriculum_state import CurriculumState, submit_episode  # noqa: E402
from CurriculumRL.logic.orientation import (  # noqa: E402
    desired_flange_orientation,
    tool_axis_alignment,
)
from CurriculumRL.logic.parking_state import initial_parking_state, update_parking_state  # noqa: E402
from CurriculumRL.logic.reset_state import (  # noqa: E402
    clone_reset_cache,
    commit_target_readback,
    create_reset_cache,
    prepare_target_reset,
)
from CurriculumRL.logic.tcp_kinematics import (  # noqa: E402
    flange_to_tcp_state,
    world_to_root_position,
    world_to_root_vector,
)
from CurriculumRL.logic.terminations import (  # noqa: E402
    illegal_contact,
    outside_workspace,
    target_disturbed,
    timed_out,
)

DTYPE = torch.float64


def tensor(value: object) -> torch.Tensor:
    return torch.tensor(value, dtype=DTYPE)


def test_tcp_state_includes_rotated_offset_and_tangential_velocity() -> None:
    # 绕 z 轴旋转 90°：F 系 +x 偏置变为 W 系 +y；omega×r 指向 -x。
    tcp_position, tcp_velocity = flange_to_tcp_state(
        tensor([[1.0, 2.0, 3.0]]),
        tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]]),
        tensor([[0.1, 0.2, 0.3]]),
        tensor([[0.0, 0.0, 2.0]]),
        tensor([1.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(tcp_position, tensor([[1.0, 3.0, 3.0]]))
    torch.testing.assert_close(tcp_velocity, tensor([[-1.9, 0.2, 0.3]]))


def test_world_to_root_is_invariant_to_environment_origin() -> None:
    root_quaternion = tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]]).repeat(2, 1)
    root_position = tensor([[10.0, 20.0, 0.0], [-7.0, 4.0, 0.0]])
    position = root_position + tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    expected = tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    torch.testing.assert_close(world_to_root_position(position, root_position, root_quaternion), expected)
    torch.testing.assert_close(world_to_root_vector(tensor([[0.0, 1.0, 0.0]]).repeat(2, 1), root_quaternion), expected)


def test_quaternion_order_and_expected_flange_orientation() -> None:
    target_quaternion = tensor([state.rotation_wxyz for state in TARGET_STATES])
    expected = desired_flange_orientation(
        target_quaternion,
        tensor(TARGET_TO_TOOL_ROTATION_T),
        tensor(FLANGE_TO_TOOL_ROTATION_F),
    )
    assert expected.shape == (4, 4)
    torch.testing.assert_close(torch.linalg.vector_norm(expected, dim=-1), torch.ones(4, dtype=DTYPE))
    # xyzw 误用会改变 state_01 的结果，此处锁定项目的 wxyz 契约。
    torch.testing.assert_close(expected[0], tensor([-0.5, -0.5, 0.5, 0.5]), atol=1e-7, rtol=1e-7)


def test_tool_axis_alignment_returns_angle_and_cosine() -> None:
    target = tensor([[1.0, 0.0, 0.0, 0.0]])
    flange = desired_flange_orientation(
        target,
        tensor(TARGET_TO_TOOL_ROTATION_T),
        tensor(FLANGE_TO_TOOL_ROTATION_F),
    )
    angle, score = tool_axis_alignment(
        flange,
        target,
        tensor(FLANGE_TO_TOOL_ROTATION_F),
        tensor(TOOL_FORWARD_AXIS),
        tensor(TARGET_DOCKING_AXIS),
    )
    torch.testing.assert_close(score, tensor([1.0]))
    torch.testing.assert_close(angle, tensor([0.0]), atol=1e-7, rtol=0.0)


def _parking_update(state, distance, speed, angle, step):
    return update_parking_state(
        state,
        tensor(distance),
        tensor(speed),
        tensor(angle),
        torch.tensor(step, dtype=torch.long),
        enter_distance_m=PARKING_ENTER_DISTANCE_M,
        exit_distance_m=PARKING_EXIT_DISTANCE_M,
        max_tcp_speed_m_s=PARKING_MAX_TCP_SPEED_M_S,
        max_orientation_error_rad=PARKING_MAX_ORIENTATION_ERROR_RAD,
        required_dwell_steps=PARKING_DWELL_POLICY_STEPS,
    )


def test_parking_hysteresis_dwell_reset_and_duplicate_step() -> None:
    state = initial_parking_state(2)
    state = _parking_update(state, [0.04, 0.0401], [0.03, 0.0], [0.0, 0.0], [0, 0])
    assert state.inside.tolist() == [True, False]
    assert state.dwell_steps.tolist() == [1, 0]
    duplicate = _parking_update(state, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0, 0])
    assert duplicate.dwell_steps.tolist() == [1, 0]
    state = _parking_update(duplicate, [0.055, 0.04], [0.03, 0.031], [0.0, 0.0], [1, 1])
    assert state.inside.tolist() == [True, True]
    assert state.success.tolist() == [True, False]
    state = _parking_update(state, [0.0551, 0.04], [0.0, 0.0], [0.0, math.pi], [2, 2])
    assert state.inside.tolist() == [False, True]
    assert state.dwell_steps.tolist() == [0, 0]


def test_partial_target_reset_does_not_modify_unselected_environments() -> None:
    cache = create_reset_cache(4)
    cache.target_position_w[:] = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    cache.best_distance_m[:] = 7.0
    before = clone_reset_cache(cache)
    env_ids = torch.tensor([1, 3])
    state_indices = torch.tensor([2, 0])
    origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    positions = torch.tensor([state.position_e for state in TARGET_STATES])
    quaternions = torch.tensor([state.rotation_wxyz for state in TARGET_STATES])
    prepositions = torch.tensor([state.preposition_e for state in TARGET_STATES])
    pose, velocity = prepare_target_reset(
        cache,
        env_ids,
        state_indices,
        origins,
        positions,
        quaternions,
        prepositions,
        torch.tensor(TARGET_TO_TOOL_ROTATION_T),
        torch.tensor(FLANGE_TO_TOOL_ROTATION_F),
    )
    assert pose.shape == (2, 7)
    assert velocity.shape == (2, 6)
    torch.testing.assert_close(pose[:, :3], positions[state_indices] + origins[env_ids])
    for unselected in (0, 2):
        for field_name in vars(cache):
            torch.testing.assert_close(getattr(cache, field_name)[unselected], getattr(before, field_name)[unselected])
    readback_position = pose[:, :3] + 0.001
    commit_target_readback(
        cache,
        env_ids,
        readback_position,
        pose[:, 3:],
        torch.tensor(TARGET_TO_TOOL_ROTATION_T),
        torch.tensor(FLANGE_TO_TOOL_ROTATION_F),
    )
    torch.testing.assert_close(cache.target_baseline_position_w[env_ids], readback_position)


def test_contact_shape_base_exclusion_and_environment_isolation() -> None:
    forces = torch.zeros((2, 3, 3))
    forces[0, 0, 0] = 100.0  # Base_Link 被排除。
    forces[1, 2, 1] = 51.0
    result = illegal_contact(forces, ("Base_Link", "Link1", "Flange"), ("Base_Link",), 50.0)
    assert result.tolist() == [False, True]
    with pytest.raises(ValueError, match="env, body, 3"):
        illegal_contact(forces.transpose(0, 1), ("Base_Link", "Link1", "Flange"), ("Base_Link",), 50.0)


def test_failure_conditions_are_vectorized() -> None:
    workspace = tensor([[-0.75, 0.75], [-0.75, 0.75], [0.20, 1.10]])
    assert outside_workspace(tensor([[0.0, 0.0, 0.2], [0.751, 0.0, 0.5]]), workspace).tolist() == [False, True]
    assert target_disturbed(
        tensor([[0.0, 0.0, 0.0], [0.031, 0.0, 0.0]]),
        tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        tensor([[0.0, 0.0, 0.051], [0.0, 0.0, 0.0]]),
        0.03,
        0.05,
    ).tolist() == [True, True]
    assert timed_out(torch.tensor([159, 160]), 160).tolist() == [False, True]


def test_curriculum_statistics_promote_and_demote_without_changing_task_thresholds() -> None:
    state = CurriculumState()
    for _ in range(4):
        state = submit_episode(
            state,
            success=True,
            safety_failure=False,
            min_episodes=4,
            promote_success_rate=0.75,
            max_promote_safety_failure_rate=0.1,
            demote_success_rate=0.25,
            max_level=4,
        )
    assert state.level == 1 and state.transition_reason == "promoted"
    for _ in range(4):
        state = submit_episode(
            state,
            success=False,
            safety_failure=False,
            min_episodes=4,
            promote_success_rate=0.75,
            max_promote_safety_failure_rate=0.1,
            demote_success_rate=0.25,
            max_level=4,
        )
    assert state.level == 0 and state.transition_reason == "demoted"


def test_project_tcp_offset_remains_single_config_value() -> None:
    assert FLANGE_TO_TCP_TRANSLATION_F == (0.0, -0.12, 0.102)
