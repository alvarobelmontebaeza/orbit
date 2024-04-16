from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


"""
Root Penalties
"""
def body_ang_acc_l2(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_ang_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)

"""
Feet Penalties
"""
def stumble(env: RLTaskEnv, factor: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    # compute the violation
    violation = torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :2], dim=2) > (factor * net_contact_forces[:, sensor_cfg.body_ids, 2])
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1)

def feet_contact_force(env: RLTaskEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculates the penalty for feet contact force exceeding a given threshold.

    Args:
        env (RLTaskEnv): The RLTaskEnv object representing the environment.
        threshold (float): The threshold value for the contact force.
        sensor_cfg (SceneEntityCfg): The configuration of the contact sensor.

    Returns:
        torch.Tensor: The penalty for exceeding the contact force threshold.
    """

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    penalty = torch.max(torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :], dim=2) - threshold)
    return torch.sum(penalty, dim=1)

def feet_flat_orientation_l2(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat feet orientation using L2-kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    feet_state = asset.data.body_state_w[:, asset_cfg.body_ids, :]
    feet_pos = feet_state[:, :, :3]
    feet_quat = feet_state[:, :, 3:7]
    '''
    feet_lin_vel = feet_state[:, :, 7:10]
    feet_ang_vel = feet_state[:, :, 10:13]
    '''

    return torch.sum(torch.sum(torch.square(feet_quat[:,:,:2]), dim=2), dim=1)


"""
Joint Penalties
"""

def joint_power_l2(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Calculates the L2 norm of the joint power for a given environment and asset configuration.

    Args:
        env (RLTaskEnv): The RLTaskEnv object representing the environment.
        asset_cfg (SceneEntityCfg, optional): The SceneEntityCfg object representing the asset configuration. Defaults to SceneEntityCfg("robot").

    Returns:
        torch.Tensor: The L2 norm of the joint power.

    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    power = torch.clamp(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids], min=0.0)
    return torch.sum(torch.square(power), dim=1)

"""
Position Command Tracking Rewards
"""

def heading_tracking_reward(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Calculates the reward for heading tracking in a reinforcement learning task environment.

    Args:
        env (RLTaskEnv): The reinforcement learning task environment.
        command_name (str): The name of the command.
        asset_cfg (SceneEntityCfg): The configuration of the scene entity.

    Returns:
        torch.Tensor: The reward for heading tracking.

    """
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name) # Assuming 2D Pose Command, shape is (num_envs, 4), where 4 is (x, y, z, heading)
    # obtain the desired heading direction
    des_heading_w = command[:, 3]
    # compute the current heading direction
    curr_heading_w = torch.atan2(asset.data.root_quat_w[:, 1], asset.data.root_quat_w[:, 0])

    return (1.0 - 0.5 * torch.norm(curr_heading_w - des_heading_w, dim=1))
