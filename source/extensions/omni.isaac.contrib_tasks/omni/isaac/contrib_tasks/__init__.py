# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module containing environments contributed by the community.


We use OpenAI Gym registry to register the environment and their default configuration file.
The default configuration file is passed to the argument "kwargs" in the Gym specification registry.
The string is parsed into respective configuration container which needs to be passed to the environment
class. This is done using the function :meth:`load_cfg_from_registry` in the sub-module
:mod:`omni.isaac.orbit.utils.parse_cfg`.

Note:
    This is a slight abuse of kwargs since they are meant to be directly passed into the environment class.
    Instead, we remove the key :obj:`cfg_file` from the "kwargs" dictionary and the user needs to provide
    the kwarg argument :obj:`cfg` while creating the environment.

Usage:
    >>> import gymnasium as gym
    >>> import omni.isaac.contrib_tasks
    >>> from omni.isaac.orbit_tasks.utils.parse_cfg import load_cfg_from_registry
    >>>
    >>> task_name = "Isaac-Contrib-<my-registered-env-name>-v0"
    >>> cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    >>> env = gym.make(task_name, cfg=cfg)
"""

from __future__ import annotations

import gymnasium as gym  # noqa: F401
import os
import toml

# Conveniences to other module directories via relative paths
ORBIT_CONTRIB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

ORBIT_CONTRIB_TASKS_METADATA = toml.load(os.path.join(ORBIT_CONTRIB_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ORBIT_CONTRIB_TASKS_METADATA["package"]["version"]

##
# Register Gym environments.
##


import gymnasium as gym

from .locomotion.velocity.config.tako import agents as vel_agents
from .locomotion.velocity.config.tako import flat_env_cfg as vel_flat_env_cfg
from .locomotion.velocity.config.tako import rough_env_cfg as vel_rough_env_cfg

from .locomotion.position.config.tako import agents as pos_agents
from .locomotion.position.config.tako import rough_env_cfg as pos_rough_env_cfg
from .locomotion.position.config.tako import flat_env_cfg as pos_flat_env_cfg

from .leg_position_control.config.tako import agents as leg_pos_agents
from .leg_position_control.config.tako import joint_torque_env_cfg as leg_pos_joint_torque_env_cfg

##
# Register Gym environments.
##
'''
gym.register(
    id="Isaac-Contrib-<my-awesome-env>-v0",
    entry_point="omni.isaac.contrib_tasks.<your-env-package>:<your-env-class>",
    disable_env_checker=True,
    kwargs={"cfg_entry_point": "omni.isaac.contrib_tasks.<your-env-package-cfg>:<your-env-class-cfg>"},
)
'''
##
# VELOCITY TASK ENVIRONMENTS
##
gym.register(
    id="Isaac-Contrib-Velocity-Flat-Tako-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": vel_flat_env_cfg.TakoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": vel_agents.rsl_rl_cfg.TakoFlatPPORunnerCfg,
        "skrl_cfg_entry_point": "omni.isaac.contrib_tasks.locomotion.velocity.config.tako.agents:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Contrib-Velocity-Flat-Tako-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": vel_flat_env_cfg.TakoFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": vel_agents.rsl_rl_cfg.TakoFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Contrib-Velocity-Rough-Tako-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": vel_rough_env_cfg.TakoRoughEnvCfg,
        "rsl_rl_cfg_entry_point": vel_agents.rsl_rl_cfg.TakoRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Contrib-Velocity-Rough-Tako-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": vel_rough_env_cfg.TakoRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": vel_agents.rsl_rl_cfg.TakoRoughPPORunnerCfg,
    },
)

##
# POSITION TASK ENVIRONMENTS
##
gym.register(
    id="Isaac-Contrib-Position-Flat-Tako-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pos_flat_env_cfg.TakoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": pos_agents.rsl_rl_cfg.TakoFlatPPORunnerCfg,
        "skrl_cfg_entry_point": "omni.isaac.contrib_tasks.locomotion.position.config.tako.agents:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Contrib-Position-Flat-Tako-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pos_flat_env_cfg.TakoFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pos_agents.rsl_rl_cfg.TakoFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Contrib-Position-Rough-Tako-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pos_rough_env_cfg.TakoRoughEnvCfg,
        "rsl_rl_cfg_entry_point": pos_agents.rsl_rl_cfg.TakoRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Contrib-Position-Rough-Tako-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pos_rough_env_cfg.TakoRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pos_agents.rsl_rl_cfg.TakoRoughPPORunnerCfg,
    },
)

##
#   LEG POSITION CONTROL TASK ENVIRONMENTS
##

gym.register(
    id="Isaac-Contrib-LegPos-Torque-Tako-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": leg_pos_joint_torque_env_cfg.TakoLegPositionControlEnvCfg,
        "rsl_rl_cfg_entry_point": leg_pos_agents.rsl_rl_cfg.TakoPPORunnerCfg,
    },
)