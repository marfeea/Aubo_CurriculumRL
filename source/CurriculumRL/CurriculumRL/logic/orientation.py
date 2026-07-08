"""TCP 工具姿态和目标停靠轴的纯张量计算。"""

from __future__ import annotations

import torch

from .tcp_kinematics import quaternion_conjugate, quaternion_multiply, rotate_vector


def desired_flange_orientation(
    target_quaternion_wxyz: torch.Tensor,
    target_to_tool_rotation_t: torch.Tensor,
    flange_to_tool_rotation_f: torch.Tensor,
) -> torch.Tensor:
    """由目标姿态和工具标定计算期望法兰世界姿态。"""

    desired_tool_wxyz = quaternion_multiply(target_quaternion_wxyz, target_to_tool_rotation_t)
    return quaternion_multiply(desired_tool_wxyz, quaternion_conjugate(flange_to_tool_rotation_f))


def tool_axis_alignment(
    flange_quaternion_wxyz: torch.Tensor,
    target_quaternion_wxyz: torch.Tensor,
    flange_to_tool_rotation_f: torch.Tensor,
    tool_forward_axis_t: torch.Tensor,
    target_docking_axis_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回工具前向轴与目标停靠轴的夹角（弧度）和余弦对齐分数。"""

    tool_quaternion_wxyz = quaternion_multiply(flange_quaternion_wxyz, flange_to_tool_rotation_f)
    tool_axis_w = rotate_vector(tool_quaternion_wxyz, tool_forward_axis_t)
    target_axis_w = rotate_vector(target_quaternion_wxyz, target_docking_axis_t)
    tool_axis_w = torch.nn.functional.normalize(tool_axis_w, dim=-1)
    target_axis_w = torch.nn.functional.normalize(target_axis_w, dim=-1)
    score = torch.sum(tool_axis_w * target_axis_w, dim=-1).clamp(-1.0, 1.0)
    return torch.acos(score), score
