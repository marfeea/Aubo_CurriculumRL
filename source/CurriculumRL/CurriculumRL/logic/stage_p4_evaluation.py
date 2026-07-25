"""P4 阶段 1 学习门禁的进程无关统计与路径簇判定。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


P4_STAGE1_EVALUATION_SCHEMA_VERSION = "p4-stage1-evaluation-v2"
P4_STAGE1_POLICY_KINDS = ("zero", "random", "ppo")


@dataclass(frozen=True, order=True)
class Stage1EvaluationUnit:
    """一个固定策略条件、随机种子和路径模式的独立 SimulationApp 单元。"""

    policy_kind: str
    seed: int
    path_mode_index: int


def stage1_evaluation_units(
    policy_kind: str, seeds: Iterable[int], path_mode_count: int
) -> tuple[Stage1EvaluationUnit, ...]:
    """建立 P4 的 ``policy × seed × z`` 固定评估矩阵。"""

    if policy_kind not in P4_STAGE1_POLICY_KINDS:
        raise ValueError(f"未知 P4 策略条件：{policy_kind}")
    if path_mode_count <= 0:
        raise ValueError("路径模式数量必须为正")
    unique_seeds = tuple(dict.fromkeys(seeds))
    if not unique_seeds:
        raise ValueError("至少提供一个固定随机种子")
    return tuple(
        Stage1EvaluationUnit(policy_kind, seed, mode)
        for mode in range(path_mode_count)
        for seed in unique_seeds
    )


def unit_result_filename(unit: Stage1EvaluationUnit) -> str:
    return f"{unit.policy_kind}_seed_{unit.seed}_path_{unit.path_mode_index}.json"


def resample_tcp_trajectory(trajectory: Sequence[Sequence[float]], point_count: int) -> np.ndarray:
    """按归一化时间重采样 TCP 轨迹，供不同 episode 长度的比较。"""

    points = np.asarray(trajectory, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("轨迹必须是至少两个有限的三维 TCP 点")
    if point_count < 2:
        raise ValueError("重采样点数必须至少为 2")
    source = np.linspace(0.0, 1.0, len(points))
    target = np.linspace(0.0, 1.0, point_count)
    return np.stack([np.interp(target, source, points[:, axis]) for axis in range(3)], axis=-1)


def trajectory_cluster_labels(
    trajectories: Iterable[Sequence[Sequence[float]]],
    *,
    point_count: int,
    mean_distance_threshold_m: float,
) -> tuple[int, ...]:
    """用重采样后平均点距作确定性贪心聚类，阈值由 P4 命令显式记录。"""

    if mean_distance_threshold_m <= 0.0:
        raise ValueError("路径簇距离阈值必须为正")
    labels: list[int] = []
    representatives: list[np.ndarray] = []
    for trajectory in trajectories:
        sample = resample_tcp_trajectory(trajectory, point_count)
        for label, representative in enumerate(representatives):
            distance = float(np.linalg.norm(sample - representative, axis=1).mean())
            if distance <= mean_distance_threshold_m:
                labels.append(label)
                break
        else:
            labels.append(len(representatives))
            representatives.append(sample)
    return tuple(labels)


def cluster_summary(labels: Sequence[int]) -> dict[str, int]:
    """返回稳定、可序列化的路径簇样本数。"""

    counts: dict[str, int] = {}
    for label in labels:
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _vector_stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "mean_abs": np.abs(values).mean(axis=0).tolist(),
        "max_abs": np.abs(values).max(axis=0).tolist(),
    }


def _mean_direction_alignment(vectors: np.ndarray, targets: np.ndarray) -> tuple[float | None, float | None]:
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(targets, axis=1)
    valid = norms > 1.0e-9
    if not valid.any():
        return None, None
    alignment = np.sum(vectors[valid] * targets[valid], axis=1) / norms[valid]
    return float(alignment.mean()), float((alignment <= 0.0).mean())


def summarize_episode_behavior(
    *,
    raw_actions: Sequence[Sequence[float]],
    processed_actions: Sequence[Sequence[float]],
    tcp_positions_b: Sequence[Sequence[float]],
    position_error_vectors_b: Sequence[Sequence[float]],
    position_errors_m: Sequence[float],
    tcp_speeds_m_s: Sequence[float],
    path_reference_distances_m: Sequence[float],
    path_reference_reached: Sequence[bool],
    controller_protections: Mapping[str, Sequence[bool]],
    terminal_position_error_m: float,
    terminal_tcp_speed_m_s: float,
    max_position_error_m: float,
    max_tcp_speed_m_s: float,
    stationary_threshold_m: float = 1.0e-4,
) -> dict[str, object]:
    """汇总一个完整 episode 的动作、运动、门槛与控制保护行为。"""

    raw = np.asarray(raw_actions, dtype=np.float64)
    processed = np.asarray(processed_actions, dtype=np.float64)
    positions = np.asarray(tcp_positions_b, dtype=np.float64)
    error_vectors = np.asarray(position_error_vectors_b, dtype=np.float64)
    errors = np.asarray(position_errors_m, dtype=np.float64)
    speeds = np.asarray(tcp_speeds_m_s, dtype=np.float64)
    reference_distances = np.asarray(path_reference_distances_m, dtype=np.float64)
    reference_reached = np.asarray(path_reference_reached, dtype=bool)
    step_count = len(errors)
    if step_count == 0:
        raise ValueError("行为汇总至少需要一个策略步")
    if raw.shape != (step_count, 6) or processed.shape != (step_count, 6):
        raise ValueError("原始动作与处理后动作必须为 (step, 6)")
    if positions.shape != (step_count, 3):
        raise ValueError("TCP 轨迹必须包含每个策略步执行前的三维位置")
    if error_vectors.shape != (step_count, 3):
        raise ValueError("位置误差向量必须为 (step, 3)")
    if any(values.shape != (step_count,) for values in (speeds, reference_distances, reference_reached)):
        raise ValueError("逐步距离、速度与参考点状态长度必须一致")
    if min(max_position_error_m, max_tcp_speed_m_s, stationary_threshold_m) <= 0.0:
        raise ValueError("行为汇总阈值必须为正数")
    numeric = (raw, processed, positions, error_vectors, errors, speeds, reference_distances)
    if not all(np.isfinite(values).all() for values in numeric):
        raise ValueError("行为汇总输入必须全部有限")
    if not np.isfinite((terminal_position_error_m, terminal_tcp_speed_m_s)).all():
        raise ValueError("终局距离与速度必须有限")

    movement = np.diff(positions, axis=0)
    movement_norm = np.linalg.norm(movement, axis=1)
    qualified = reference_reached & (errors <= max_position_error_m) & (speeds <= max_tcp_speed_m_s)
    reached_indices = np.flatnonzero(reference_reached)
    errors_with_terminal = np.append(errors, terminal_position_error_m)
    closest_index = int(np.argmin(errors_with_terminal))
    action_alignment, action_nonpositive_fraction = _mean_direction_alignment(processed[:, :3], error_vectors)
    motion_alignment, motion_nonpositive_fraction = _mean_direction_alignment(movement, error_vectors[:-1])
    protection_rates: dict[str, float] = {}
    for name, values in controller_protections.items():
        flags = np.asarray(values, dtype=bool)
        if flags.shape != (step_count,):
            raise ValueError(f"控制保护 {name} 的长度与策略步数不一致")
        protection_rates[name] = float(flags.mean())

    return {
        "policy_steps": step_count,
        "initial_position_error_m": float(errors[0]),
        "minimum_position_error_m": float(errors_with_terminal.min()),
        "minimum_position_error_step": min(closest_index + 1, step_count),
        "final_position_error_m": float(terminal_position_error_m),
        "final_tcp_speed_m_s": float(terminal_tcp_speed_m_s),
        "regression_after_minimum_m": float(terminal_position_error_m - errors_with_terminal.min()),
        "initial_path_reference_distance_m": float(reference_distances[0]),
        "minimum_path_reference_distance_m": float(reference_distances.min()),
        "final_path_reference_distance_m": float(reference_distances[-1]),
        "path_reference_reached_ever": bool(reference_reached.any()),
        "first_path_reference_reached_step": int(reached_indices[0]) + 1 if len(reached_indices) else None,
        "stage_qualified_step_count": int(qualified.sum()),
        "stage_qualified_longest_run": _longest_true_run(qualified),
        "tcp_path_length_m": float(movement_norm.sum()),
        "net_tcp_displacement_m": float(np.linalg.norm(positions[-1] - positions[0])),
        "stationary_step_fraction": float((movement_norm <= stationary_threshold_m).mean()),
        "mean_tcp_speed_m_s": float(speeds.mean()),
        "max_tcp_speed_m_s": float(speeds.max()),
        "raw_action": _vector_stats(raw),
        "processed_action": _vector_stats(processed),
        "translation_action_target_alignment_mean": action_alignment,
        "translation_action_nonpositive_alignment_fraction": action_nonpositive_fraction,
        "tcp_motion_target_alignment_mean": motion_alignment,
        "tcp_motion_nonpositive_alignment_fraction": motion_nonpositive_fraction,
        "controller_protection_rate_per_policy_step": protection_rates,
    }
