# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Script to train RL agent with Stable Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import json
import re
import signal
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _bootstrap import add_package_source  # noqa: E402

add_package_source()

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--rollout-steps", type=int, default=None, help="Override PPO rollout length for smoke runs.")
parser.add_argument("--run-tag", type=str, default="baseline", help="Stable label included in logs and checkpoints.")
parser.add_argument("--curriculum-level", type=int, default=0, help="阶段 E 的初始课程等级。")
parser.add_argument(
    "--curriculum-state", type=Path, default=None, help="恢复训练时的课程状态 JSON；默认从 checkpoint 推导。"
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import logging
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)
import CurriculumRL.tasks  # noqa: F401


def _vecnormalize_path(checkpoint: Path) -> Path:
    periodic_match = re.fullmatch(r"(.+_model)_(\d+_steps)", checkpoint.stem)
    if periodic_match:
        name = f"{periodic_match.group(1)}_vecnormalize_{periodic_match.group(2)}.pkl"
    else:
        name = f"{checkpoint.stem}_vecnormalize.pkl"
    return checkpoint.with_name(name)


def _curriculum_state_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.stem}_curriculum.json")


class CurriculumStateCheckpointCallback(BaseCallback):
    """让课程窗口与每个 SB3 checkpoint 使用同一 timestep 命名。"""

    def __init__(self, base_env: object, *, save_freq: int, save_path: str, name_prefix: str) -> None:
        super().__init__()
        self.base_env = base_env
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            from CurriculumRL.tasks.tcp_docking.mdp.runtime_state import export_curriculum_state

            output = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps_curriculum.json"
            output.write_text(
                json.dumps(export_curriculum_state(self.base_env), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args_cli.run_tag):
        raise ValueError("--run-tag 只能包含字母、数字、点、下划线和连字符")
    if args_cli.task == "CurriculumRL-TcpDocking-v0" and not 0 <= args_cli.curriculum_level <= 4:
        raise ValueError("--curriculum-level 必须位于阶段 E 的 [0, 4] 范围")
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    curriculum_resume_path: Path | None = None
    curriculum_resume_snapshot: dict[str, object] | None = None
    if args_cli.task == "CurriculumRL-TcpDocking-v0" and args_cli.checkpoint is not None:
        curriculum_resume_path = args_cli.curriculum_state or _curriculum_state_path(Path(args_cli.checkpoint))
        if not curriculum_resume_path.is_file():
            raise FileNotFoundError(f"恢复阶段 E 训练缺少课程状态：{curriculum_resume_path}")
        curriculum_resume_snapshot = json.loads(curriculum_resume_path.read_text(encoding="utf-8"))

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    if args_cli.rollout_steps is not None:
        if args_cli.rollout_steps < 2:
            raise ValueError("PPO --rollout-steps 必须至少为 2")
        agent_cfg["n_steps"] = args_cli.rollout_steps
        agent_cfg["batch_size"] = min(agent_cfg["batch_size"], args_cli.rollout_steps * env_cfg.scene.num_envs)
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.task == "CurriculumRL-TcpDocking-v0":
        env_cfg.curriculum_enabled = True
        env_cfg.curriculum_initial_level = (
            int(curriculum_resume_snapshot["level"])
            if curriculum_resume_snapshot is not None
            else args_cli.curriculum_level
        )

    # directory for logging into
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_info = f"L{args_cli.curriculum_level}_seed{agent_cfg['seed']}_{args_cli.run_tag}_{timestamp}"
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {run_info}")
    log_dir = os.path.join(log_root_path, run_info)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(
        os.path.join(log_dir, "params", "run.yaml"),
        {
            "task": args_cli.task,
            "curriculum_level": args_cli.curriculum_level,
            "effective_curriculum_initial_level": env_cfg.curriculum_initial_level
            if args_cli.task == "CurriculumRL-TcpDocking-v0"
            else None,
            "seed": agent_cfg["seed"],
            "run_tag": args_cli.run_tag,
            "curriculum_state": "params/curriculum.json",
            "env_config_snapshot": "params/env.yaml",
            "agent_config_snapshot": "params/agent.yaml",
        },
    )

    # save command used to run the script
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command, encoding="utf-8")

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = env.unwrapped

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    norm_keys = {"normalize_input", "normalize_value", "clip_obs"}
    norm_args = {}
    for key in norm_keys:
        if key in agent_cfg:
            norm_args[key] = agent_cfg.pop(key)

    if norm_args and norm_args.get("normalize_input"):
        print(f"Normalizing input, {norm_args=}")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=norm_args["normalize_input"],
            norm_reward=norm_args.get("normalize_value", False),
            clip_obs=norm_args.get("clip_obs", 100.0),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # create agent from stable baselines
    agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
    if args_cli.checkpoint is not None:
        checkpoint = Path(args_cli.checkpoint)
        if args_cli.task == "CurriculumRL-TcpDocking-v0":
            vecnormalize_path = _vecnormalize_path(checkpoint)
            if not vecnormalize_path.is_file():
                raise FileNotFoundError(f"恢复阶段 E 训练缺少 VecNormalize 状态：{vecnormalize_path}")
            if not isinstance(env, VecNormalize):
                raise RuntimeError("阶段 E 恢复训练要求启用 VecNormalize")
            env = VecNormalize.load(vecnormalize_path, env)
            env.training = True
            from CurriculumRL.tasks.tcp_docking.mdp.runtime_state import restore_curriculum_state

            assert curriculum_resume_snapshot is not None
            restore_curriculum_state(base_env, curriculum_resume_snapshot)
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

    # callbacks for agent
    task_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", args_cli.task)
    checkpoint_prefix = f"{task_slug}_L{args_cli.curriculum_level}_seed{agent_cfg['seed']}_{args_cli.run_tag}_model"
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=log_dir,
        name_prefix=checkpoint_prefix,
        save_vecnormalize=True,
        verbose=2,
    )
    callbacks = [checkpoint_callback, LogEveryNTimesteps(n_steps=args_cli.log_interval)]
    if args_cli.task == "CurriculumRL-TcpDocking-v0":
        callbacks.append(
            CurriculumStateCheckpointCallback(
                base_env, save_freq=1000, save_path=log_dir, name_prefix=checkpoint_prefix
            )
        )

    # train the agent
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
            log_interval=None,
        )
    # save the final model
    agent.save(os.path.join(log_dir, checkpoint_prefix))
    print("Saving to:")
    print(os.path.join(log_dir, f"{checkpoint_prefix}.zip"))

    if isinstance(env, VecNormalize):
        print("Saving normalization")
        env.save(os.path.join(log_dir, f"{checkpoint_prefix}_vecnormalize.pkl"))
    if args_cli.task == "CurriculumRL-TcpDocking-v0":
        from CurriculumRL.tasks.tcp_docking.mdp.runtime_state import export_curriculum_state

        curriculum_snapshot = export_curriculum_state(base_env)
        snapshot_path = Path(log_dir) / f"{checkpoint_prefix}_curriculum.json"
        snapshot_path.write_text(json.dumps(curriculum_snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (Path(log_dir) / "params" / "curriculum.json").write_text(
            json.dumps(curriculum_snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
