# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.orbit.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import RayCaster
from omni.isaac.orbit.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv, RLTaskEnv

"""
Root state.
"""

"""
Joint state.
"""

"""
Sensors.
"""

def feet_contacts(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Determines the number of feet in contact with the ground based on a force threshold in the Z direction.

    Args:
        env (BaseEnv): The environment object.
        sensor_cfg (SceneEntityCfg): The configuration for the scene entity.

    Returns:
        torch.Tensor: A tensor indicating which feet are in contact with the ground.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check the number of feet in contact with the ground by using a force threshold in the Z direction
    feet_in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    return torch.sum(feet_in_contact, dim=1)

"""
Actions.
"""
def last_processed_action(env: BaseEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).processed_actions

def docking_state(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    The docking state of the robot.

    Args:
        env (BaseEnv): The environment object.
        action_name (str): The name of the action term.

    Returns:
        torch.Tensor: The docking state of the robot.
    """
    # If the target position is close to the ground. we assume we want to dock. During test time, this info will be
    # provided by the planner
    asset: Articulation = env.scene[asset_cfg.name]
    desired_pos = env.command_manager.get_command(command_name)[:, :3]
    current_pos = asset.data.body_pos_w[:, asset_cfg.body_ids].view(-1, 3)
    des_pos_w = desired_pos + asset.data.root_pos_w
    # obtain the docking state
    docking_state = torch.zeros_like(desired_pos[:, 0])
    in_dock = torch.logical_and(des_pos_w[:, 2] < 0.05, torch.norm(desired_pos - current_pos, dim=1) < 0.05)
    docking_state[in_dock] = 1.0

    return docking_state


"""
Commands.
"""

def target_2d_position(env: RLTaskEnv, command_name: str) -> torch.Tensor:
    """
    Get the 2D position of the target for a given command.

    Args:
        env (RLTaskEnv): The RLTaskEnv object.
        command_name (str): The name of the command.

    Returns:
        torch.Tensor: A tensor containing the 2D position of the target.
    """

    command = env.command_manager.get_command(command_name)
    # Obtain desired 2d position
    return command[:, :2]

def target_heading(env: RLTaskEnv, command_name: str) -> torch.Tensor:
    """
    Get the target heading from the command.

    Args:
        env (RLTaskEnv): The RLTaskEnv object.
        command_name (str): The name of the command.

    Returns:
        torch.Tensor: The target heading.

    """
    command = env.command_manager.get_command(command_name)
    # obtain the desired heading
    return command[:, 3].unsqueeze(-1)

def remaining_time(env: RLTaskEnv, command_name: str) -> torch.Tensor:
    """
    Get the remaining time for the command to reach the target.

    Args:
        env (RLTaskEnv): The RLTaskEnv object.
        command_name (str): The name of the command.

    Returns:
        torch.Tensor: The remaining time for the command to reach the target.
    """

    # obtain the remaining time
    remaining_time = env.command_manager.get_term(command_name).time_left
    return remaining_time.unsqueeze(-1)