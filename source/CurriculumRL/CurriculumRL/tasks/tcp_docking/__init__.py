"""AUBO TCP 停靠任务注册。"""

import gymnasium as gym

from . import agents

gym.register(
    id="CurriculumRL-TcpDocking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TcpDockingEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
