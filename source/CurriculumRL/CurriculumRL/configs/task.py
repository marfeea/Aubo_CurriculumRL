"""TCP 标定、目标状态、最终成功条件和安全阈值的唯一事实源。"""

from dataclasses import dataclass
from math import pi, sqrt
from typing import Final


@dataclass(frozen=True)
class TargetState:
    """环境局部坐标中的离散目标状态。"""

    name: str
    position_e: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    preposition_e: tuple[float, float, float]


FLANGE_TO_TCP_TRANSLATION_F: Final = (0.0, -0.12, 0.102)
FLANGE_TO_TOOL_ROTATION_F: Final = (sqrt(0.5), 0.0, -sqrt(0.5), 0.0)
TARGET_TO_TOOL_ROTATION_T: Final = (sqrt(0.5), sqrt(0.5), 0.0, 0.0)
TOOL_FORWARD_AXIS: Final = (0.0, 0.0, 1.0)
TARGET_DOCKING_AXIS: Final = (0.0, -1.0, 0.0)

TARGET_STATES: Final = (
    TargetState("sample_bottle_state_01", (1.537, 0.203, 0.94), (0.0, 0.0, 0.0, 1.0), (1.537, 0.083, 0.94)),
    TargetState(
        "sample_bottle_state_02",
        (0.91167, 0.1753, 0.96789),
        (0.70710678, 0.0, 0.0, -0.70710678),
        (1.03167, 0.1753, 0.96789),
    ),
    TargetState(
        "sample_bottle_state_03",
        (0.91167, 0.03036, 0.96676),
        (0.70710678, 0.0, 0.0, -0.70710678),
        (1.03167, 0.03036, 0.96676),
    ),
    TargetState(
        "sample_bottle_state_04",
        (0.91235, -0.18557, 0.99091),
        (0.70710678, 0.0, 0.0, -0.70710678),
        (1.03235, -0.18557, 0.99091),
    ),
)

PARKING_ENTER_DISTANCE_M: Final = 0.04
PARKING_EXIT_DISTANCE_M: Final = 0.055
PARKING_MAX_TCP_SPEED_M_S: Final = 0.03
PARKING_MAX_ORIENTATION_ERROR_RAD: Final = 10.0 * pi / 180.0
PARKING_DWELL_POLICY_STEPS: Final = 2

TCP_WORKSPACE_B: Final = ((-0.75, 0.75), (-0.75, 0.75), (0.20, 1.10))
ILLEGAL_CONTACT_FORCE_N: Final = 50.0
MAX_TARGET_DISPLACEMENT_M: Final = 0.03
MAX_TARGET_LINEAR_SPEED_M_S: Final = 0.05
