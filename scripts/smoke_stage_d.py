"""阶段 D V4：验证注册、有限张量、并行 reset 和奖励/终止统计。"""

from __future__ import annotations

import argparse
import traceback

from _bootstrap import add_package_source

add_package_source()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import CurriculumRL.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from CurriculumRL.configs.assets import SCENE_ENTITY_AUBO, SCENE_ENTITY_TARGET  # noqa: E402
from CurriculumRL.tasks.tcp_docking.mdp.runtime_state import compute_step  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


def _policy_observation(observation: object) -> torch.Tensor:
    if not isinstance(observation, dict) or "policy" not in observation:
        raise RuntimeError(f"策略观测必须是包含 policy 的字典，实际 {type(observation)}")
    policy = observation["policy"]
    if not torch.is_tensor(policy):
        raise RuntimeError(f"policy 观测必须是张量，实际 {type(policy)}")
    return policy


def main() -> int:
    print("[D-V4] 解析环境配置", flush=True)
    env_cfg = parse_env_cfg(
        "CurriculumRL-TcpDocking-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    print("[D-V4] 创建环境", flush=True)
    env = gym.make(
        "CurriculumRL-TcpDocking-v0",
        cfg=env_cfg,
    )
    base = env.unwrapped
    print("[D-V4] 全环境 reset", flush=True)
    observation, _ = env.reset()
    policy = _policy_observation(observation)
    if policy.shape != (args_cli.num_envs, 29) or not torch.isfinite(policy).all():
        raise RuntimeError(f"策略观测形状/有限值异常：shape={tuple(policy.shape)}")

    target_before = base.scene[SCENE_ENTITY_TARGET].data.root_state_w.clone()
    robot_before = base.scene[SCENE_ENTITY_AUBO].data.joint_pos.clone()
    selected = torch.tensor([1], device=base.device)
    print("[D-V4] 部分 reset 隔离检查", flush=True)
    base.reset(env_ids=selected)
    unselected = torch.tensor([index for index in range(args_cli.num_envs) if index != 1], device=base.device)
    torch.testing.assert_close(base.scene[SCENE_ENTITY_TARGET].data.root_state_w[unselected], target_before[unselected])
    torch.testing.assert_close(base.scene[SCENE_ENTITY_AUBO].data.joint_pos[unselected], robot_before[unselected])

    reward_min = torch.full((), torch.inf, device=base.device)
    reward_max = torch.full((), -torch.inf, device=base.device)
    terminated_count = 0
    truncated_count = 0
    print("[D-V4] 零动作/随机动作短回合", flush=True)
    for step_index in range(args_cli.steps):
        action = torch.zeros((args_cli.num_envs, 6), device=base.device)
        if step_index >= args_cli.steps // 2:
            action.uniform_(-1.0, 1.0)
        observation, reward, terminated, truncated, _ = env.step(action)
        tensors = (_policy_observation(observation), reward, terminated, truncated)
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise RuntimeError(f"第 {step_index} 步出现 NaN/Inf")
        reward_min = torch.minimum(reward_min, reward.min())
        reward_max = torch.maximum(reward_max, reward.max())
        terminated_count += int(terminated.sum().item())
        truncated_count += int(truncated.sum().item())
        last_causes = {
            name: int(base.termination_manager._last_episode_dones[:, index].sum().item())
            for index, name in enumerate(base.termination_manager.active_terms)
        }
        print(f"[D-V4] step={step_index + 1}/{args_cli.steps}, last_causes={last_causes}", flush=True)

    metrics, components = compute_step(base)
    component_summary = {name: (float(value.min()), float(value.max())) for name, value in vars(components).items()}
    termination_summary = {
        "success": int(metrics.success.sum()),
        "outside_workspace": int(metrics.outside_workspace.sum()),
        "illegal_contact": int(metrics.illegal_contact.sum()),
        "target_disturbed": int(metrics.target_disturbed.sum()),
        "terminated": terminated_count,
        "truncated": truncated_count,
    }
    print(
        f"V4 smoke: observation={tuple(policy.shape)}, action=(6,), "
        f"reward_range=({float(reward_min):.6f}, {float(reward_max):.6f}), "
        f"components={component_summary}, terminations={termination_summary}",
        flush=True,
    )
    env.close()
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
