"""P3 动作掩码、路径约束和 C1 过滤接触的纯逻辑回归。"""

import sys
from pathlib import Path

import torch

PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "source" / "CurriculumRL"
sys.path.insert(0, str(PACKAGE_SOURCE))

from CurriculumRL.configs.curriculum import CURRICULUM_COLLISION_PROFILES, CURRICULUM_V2_PATH_MODES  # noqa: E402
from CurriculumRL.logic.path_constraints import (  # noqa: E402
    initial_path_constraint_state,
    reset_path_constraint_state,
    update_path_constraint,
)
from CurriculumRL.logic.terminations import illegal_filtered_contact  # noqa: E402


def test_path_modes_produce_three_distinct_midpoint_constraints() -> None:
    state = initial_path_constraint_state(3)
    starts = torch.zeros((3, 3))
    targets = torch.tensor([[0.6, 0.0, 0.0]]).expand(3, -1)
    offsets = torch.tensor([mode.midpoint_offset_b for mode in CURRICULUM_V2_PATH_MODES])
    reset_path_constraint_state(state, torch.tensor([0, 1, 2]), starts, targets, torch.tensor([0, 1, 2]), offsets)
    torch.testing.assert_close(state.reference_point_b[0], torch.tensor([0.3, 0.0, 0.0]))
    torch.testing.assert_close(state.reference_point_b[1], torch.tensor([0.3, 0.12, 0.0]))
    torch.testing.assert_close(state.reference_point_b[2], torch.tensor([0.3, -0.12, 0.0]))


def test_path_reference_is_latched_after_reaching_it() -> None:
    state = initial_path_constraint_state(1)
    offsets = torch.tensor([mode.midpoint_offset_b for mode in CURRICULUM_V2_PATH_MODES])
    reset_path_constraint_state(
        state, torch.tensor([0]), torch.zeros((1, 3)), torch.tensor([[0.6, 0.0, 0.0]]), torch.tensor([1]), offsets
    )
    reached = update_path_constraint(state, torch.tensor([[0.3, 0.12, 0.0]]), reference_reach_tolerance_m=0.04)
    assert reached.reference_reached_now.tolist() == [True]
    moved_on = update_path_constraint(state, torch.tensor([[0.6, 0.0, 0.0]]), reference_reach_tolerance_m=0.04)
    assert moved_on.reference_reached.tolist() == [True]
    assert moved_on.reference_reached_now.tolist() == [False]


def test_c1_profile_has_only_glass_filter_and_filtered_contact_ignores_base() -> None:
    c1 = CURRICULUM_COLLISION_PROFILES[0]
    assert c1.identifier == "C1"
    assert c1.filter_prim_paths_expr == ("{ENV_REGEX_NS}/station/static/workstation/WorkStation/Glass/M_Glass",)
    forces = torch.zeros((1, 2, 1, 3))
    forces[0, 0, 0, 0] = 100.0
    forces[0, 1, 0, 0] = 60.0
    assert illegal_filtered_contact(forces, ("Base_Link", "Flange"), ("Base_Link",), 50.0).tolist() == [True]
    forces[0, 1, 0, 0] = 0.0
    assert illegal_filtered_contact(forces, ("Base_Link", "Flange"), ("Base_Link",), 50.0).tolist() == [False]
