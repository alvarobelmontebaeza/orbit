
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

"""
Tako MDP terminations.
"""

"""
Contact sensor.
"""


def illegal_contact(env: RLTaskEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )

def feet_contact_num(env: RLTaskEnv, threshold: int, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the number of feet in contact with the ground is less than the threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if the number of feet in contact with the ground is less than the threshold
    feet_in_contact = torch.sum(
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0, dim=1
    )
    return (feet_in_contact < threshold) * (env.episode_length_buf > 100)

def unhealthy_base_position(env: RLTaskEnv, max_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the base height is less than the threshold."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if the base height is less than the threshold
    return (asset.data.root_state_w[:, 2] > max_height) + (torch.abs(asset.data.projected_gravity_b[:, 2]) < 0.5)

def target_reached(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Calculates the reward for standing at the target in a reinforcement learning task environment.

    Args:
        env (RLTaskEnv): The reinforcement learning task environment.
        command_name (str): The name of the command.
        asset_cfg (SceneEntityCfg): The configuration of the scene entity.

    Returns:
        torch.Tensor: The reward for standing at the target.

    """
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name) # Assuming 2D Pose Command, shape is (num_envs, 4), where 4 is (x, y, z, heading)
    # obtain the desired position and heading
    des_pos_b = command[:, :2] # Only x and y 
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    des_heading_w = command[:, 3]
    # compute the current position and heading
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :2]  # type: ignore
    curr_heading_w = torch.atan2(asset.data.root_quat_w[:, 1], asset.data.root_quat_w[:, 0])

    # Check if target has been reached
    position_reached = torch.norm(curr_pos_w - des_pos_w, dim=1) < 0.25
    heading_reached = torch.abs(curr_heading_w - des_heading_w) < 0.5

    return position_reached * heading_reached
