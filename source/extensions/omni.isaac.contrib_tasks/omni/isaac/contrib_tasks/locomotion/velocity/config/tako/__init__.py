# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Contrib-Velocity-Flat-Tako-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TakoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.TakoFlatPPORunnerCfg,
        "skrl_cfg_entry_point": "omni.isaac.contrib_tasks.locomotion.velocity.config.tako.agents:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Contrib-Velocity-Flat-Tako-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TakoFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.TakoFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Contrib-Velocity-Rough-Tako-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.TakoRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.TakoRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Contrib-Velocity-Rough-Tako-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.TakoRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.TakoRoughPPORunnerCfg,
    },
)
