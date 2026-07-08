"""Differential IK 的动作接口契约；数值参数在阶段 C 确认。"""

from typing import Final

ACTION_DIMENSION: Final = 6
ACTION_FRAME: Final = "robot_root_B"
ROTATION_INCREMENT_REPRESENTATION: Final = "rotation_vector"
