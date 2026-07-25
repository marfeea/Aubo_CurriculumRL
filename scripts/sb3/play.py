# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import re
import sys
import threading
import traceback
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _bootstrap import add_package_source  # noqa: E402

add_package_source()

from isaaclab.app import AppLauncher

_watchdog_stop: threading.Event | None = None
_watchdog_heartbeat: threading.Event | None = None
_watchdog_thread: threading.Thread | None = None


def _dump_python_thread_stacks() -> None:
    """从普通 Python 线程中输出栈，不使用 Kit 不兼容的原生 signal 定时器。"""

    print("[REPLAY_WATCHDOG] 超时未见 Python 检查点；当前线程栈如下：", file=sys.stderr, flush=True)
    frames = sys._current_frames()
    for thread in threading.enumerate():
        frame = frames.get(thread.ident)
        if frame is None:
            continue
        print(f"[REPLAY_WATCHDOG] thread={thread.name} ident={thread.ident}", file=sys.stderr, flush=True)
        traceback.print_stack(frame, file=sys.stderr)


def _watchdog_loop(timeout_s: float) -> None:
    """在检查点长期未返回时打印 Python 栈，且不向 Omniverse Kit 注入 native timer。"""

    assert _watchdog_stop is not None
    assert _watchdog_heartbeat is not None
    while not _watchdog_stop.is_set():
        if _watchdog_heartbeat.wait(timeout_s):
            _watchdog_heartbeat.clear()
        elif not _watchdog_stop.is_set():
            _dump_python_thread_stacks()


def _checkpoint(message: str) -> None:
    """输出回放阶段边界，便于区分无响应发生在何处。"""

    if _watchdog_heartbeat is not None:
        _watchdog_heartbeat.set()
    print(f"[REPLAY_CHECKPOINT] {message}", flush=True)

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--episodes", type=int, default=None, help="完成指定数量的完整 episode 后退出。")
parser.add_argument(
    "--render-interval",
    type=int,
    default=None,
    help="可选：每隔多少物理步刷新一次渲染；仅影响渲染，不改变物理或策略控制频率。",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--curriculum-state", type=Path, default=None, help="TCP 停靠回放的课程状态 JSON；默认从 checkpoint 推导。"
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--steps", type=int, default=None, help="Exit after this many policy steps.")
parser.add_argument(
    "--debug-traceback-timeout-s",
    type=float,
    default=0.0,
    help="大于 0 时，若该秒数内未返回 Python 解释器则重复打印所有线程栈；默认关闭。",
)
parser.add_argument(
    "--debug-progress-interval",
    type=int,
    default=0,
    help="大于 0 时，每 N 个策略步输出一次回放进度；默认关闭。",
)
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.episodes is not None and args_cli.episodes <= 0:
    parser.error("--episodes 必须为正整数")
if args_cli.render_interval is not None and args_cli.render_interval <= 0:
    parser.error("--render-interval 必须为正整数")
if args_cli.debug_traceback_timeout_s < 0.0:
    parser.error("--debug-traceback-timeout-s 不能为负数")
if args_cli.debug_progress_interval < 0:
    parser.error("--debug-progress-interval 不能为负数")
if args_cli.debug_traceback_timeout_s > 0.0:
    _watchdog_stop = threading.Event()
    _watchdog_heartbeat = threading.Event()
    _watchdog_thread = threading.Thread(
        target=_watchdog_loop,
        args=(args_cli.debug_traceback_timeout_s,),
        name="replay-python-watchdog",
        daemon=True,
    )
    _watchdog_thread.start()
    _checkpoint(f"已启用 {args_cli.debug_traceback_timeout_s:g} 秒 Python watchdog traceback")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import CurriculumRL.tasks  # noqa: F401
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with stable-baselines agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.render_interval is not None:
        env_cfg.sim.render_interval = args_cli.render_interval
    _checkpoint(
        f"配置完成：task={args_cli.task}, device={env_cfg.sim.device}, num_envs={env_cfg.scene.num_envs}, "
        f"render_interval={env_cfg.sim.render_interval}"
    )

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", train_task_name)
    log_root_path = os.path.abspath(log_root_path)
    # checkpoint and log_dir stuff
    if args_cli.use_pretrained_checkpoint:
        checkpoint_path = get_published_pretrained_checkpoint("sb3", train_task_name)
        if not checkpoint_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # FIXME: last checkpoint doesn't seem to really use the last one'
        if args_cli.use_last_checkpoint:
            checkpoint = ".*_model_.*_steps.zip"
        else:
            checkpoint = ".*_model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint, sort_alpha=False)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)
    checkpoint_path = Path(checkpoint_path)

    curriculum_snapshot: dict[str, object] | None = None
    if args_cli.task == "CurriculumRL-TcpDocking-v0":
        from CurriculumRL.configs.curriculum import CURRICULUM_CONFIG_VERSION
        from CurriculumRL.logic.stage_e_evaluation import curriculum_state_path, validate_curriculum_snapshot

        snapshot_path = args_cli.curriculum_state or curriculum_state_path(checkpoint_path)
        if not snapshot_path.is_file():
            raise FileNotFoundError(f"TCP 停靠回放缺少课程状态：{snapshot_path}")
        curriculum_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        validate_curriculum_snapshot(curriculum_snapshot, CURRICULUM_CONFIG_VERSION)
        env_cfg.curriculum_initial_level = int(curriculum_snapshot["level"])
        _checkpoint(f"课程快照已校验：{snapshot_path}")

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    _checkpoint("gym 环境已创建")
    base_env = env.unwrapped
    if curriculum_snapshot is not None:
        from CurriculumRL.tasks.tcp_docking.mdp.runtime_state import restore_curriculum_state

        restore_curriculum_state(base_env, curriculum_snapshot)
        _checkpoint("课程状态已恢复到环境")

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        _checkpoint("录像包装器已创建")
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)
    _checkpoint("SB3 向量环境包装完成")

    periodic_match = re.fullmatch(r"(.+_model)_(\d+_steps)", checkpoint_path.stem)
    if periodic_match:
        vec_norm_name = f"{periodic_match.group(1)}_vecnormalize_{periodic_match.group(2)}.pkl"
    else:
        vec_norm_name = f"{checkpoint_path.stem}_vecnormalize.pkl"
    vec_norm_path = checkpoint_path.with_name(vec_norm_name)

    # normalize environment (if needed)
    if vec_norm_path.exists():
        print(f"Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        #  do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False
        _checkpoint("VecNormalize 已加载并冻结")
    elif "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
        )

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = PPO.load(checkpoint_path, env, print_system_info=True)
    _checkpoint("PPO checkpoint 已加载")

    dt = env.unwrapped.step_dt

    # reset environment
    _checkpoint("开始 env.reset()")
    obs = env.reset()
    _checkpoint("env.reset() 完成")
    timestep = 0
    completed_episodes = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        debug_this_step = args_cli.debug_progress_interval and timestep % args_cli.debug_progress_interval == 0
        if debug_this_step:
            _checkpoint(f"开始策略步={timestep + 1} 的 PPO action")
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, _ = agent.predict(obs, deterministic=True)
            if debug_this_step:
                _checkpoint(f"策略步={timestep + 1} 的 PPO action 已生成")
            # env stepping
            if debug_this_step:
                _checkpoint(f"开始策略步={timestep + 1} 的 env.step()")
            obs, _, dones, _ = env.step(actions)
            if debug_this_step:
                _checkpoint(f"策略步={timestep + 1} 的 env.step() 已完成")
        timestep += 1
        completed_episodes += int(dones.sum())
        if args_cli.debug_progress_interval and timestep % args_cli.debug_progress_interval == 0:
            _checkpoint(f"策略步={timestep}, completed_episodes={completed_episodes}")
        if args_cli.episodes is not None and completed_episodes >= args_cli.episodes:
            break
        if args_cli.video and args_cli.episodes is None:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.steps is not None and timestep >= args_cli.steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        _checkpoint("捕获到未处理异常，以下为完整 traceback")
        traceback.print_exc()
        raise
    finally:
        if _watchdog_stop is not None:
            _watchdog_stop.set()
        if _watchdog_heartbeat is not None:
            _watchdog_heartbeat.set()
        if _watchdog_thread is not None:
            _watchdog_thread.join(timeout=1.0)
        simulation_app.close()
