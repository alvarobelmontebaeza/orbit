from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.utils.math import combine_frame_transforms, quat_mul, quat_error_magnitude

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


"""
Root Penalties
"""
def body_ang_acc_l2(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_ang_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)

def action_term_l2(env: RLTaskEnv, action_name: str) -> torch.Tensor:
    """Penalize the actions using L2-kernel."""
    return torch.sum(torch.square(env.action_manager.get_term(action_name).processed_actions), dim=1)

def action_term_rate_l2(env: RLTaskEnv, term_name: str) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2-kernel."""
    action_term_dim = env.action_manager.get_term(term_name).action_dim
    return torch.sum(torch.square(env.action_manager.action[:, :action_term_dim] - env.action_manager.prev_action[:, :action_term_dim]), dim=1)



"""
Feet Penalties
"""
def stumble(env: RLTaskEnv, factor: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    # compute the violation
    violation = torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :2], dim=2)
    return torch.sum(violation, dim=1)

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
    penalty = torch.sum(torch.max(torch.square(torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :]) - threshold)), dim=1)
    return penalty

def feet_xy_vel_in_contact(env: RLTaskEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculates the penalty for feet contact velocity exceeding a given threshold.

    Args:
        env (RLTaskEnv): The RLTaskEnv object representing the environment.
        threshold (float): The threshold value for the contact velocity.
        sensor_cfg (SceneEntityCfg): The configuration of the contact sensor.

    Returns:
        torch.Tensor: The penalty for exceeding the contact velocity threshold.
    """

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_vel_xy = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :2]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    penalty = torch.sum(torch.norm(feet_vel_xy, dim=2) * in_contact, dim=1) 

    return penalty

def feet_power(env: RLTaskEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculates the penalty for feet power exceeding a given threshold.

    Args:
        env (RLTaskEnv): The RLTaskEnv object representing the environment.
        threshold (float): The threshold value for the contact power.
        sensor_cfg (SceneEntityCfg): The configuration of the contact sensor.

    Returns:
        torch.Tensor: The penalty for exceeding the contact power threshold.
    """

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    feet_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids]
    feet_contact_forces = net_contact_forces[:, sensor_cfg.body_ids]

    feet_power = torch.norm(feet_vel * feet_contact_forces, dim=2)

    return torch.sum(feet_power, dim=1)



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
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(power), dim=1)

"""
Position Command Tracking Rewards
"""

def position_tracking_reward(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.root_pos_w
    return (1.0 - 0.5 * torch.norm(curr_pos_w[:] - des_pos_w[:], dim=1)) #* (env.command_manager.get_term(command_name).time_left < 1.0)

def base_pose_tracking_reward(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sigma: float = 2.0) -> torch.Tensor:
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_orient_b = command[:, 3:7]
    des_pos_w = des_pos_b + torch.tensor(asset.cfg.init_state.pos, device=env.device) + env.scene.env_origins #combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b) #des_pos_b + env.scene.env_origins
    des_orient_w = quat_mul(asset.data.root_state_w[:, 3:7], des_orient_b)
    curr_pos_w = asset.data.root_pos_w
    curr_orient_w = asset.data.root_quat_w

    pos_error = torch.norm(des_pos_b, dim=1) # Since command is UniformPose3dCommand, the desired position in body frame is the position error
    orient_error = quat_error_magnitude(curr_orient_w, des_orient_w)
    pos_tracking_rew = -torch.log(1.e-5 + pos_error**2) 
    #pos_tracking_rew = torch.exp(-(1.0/sigma) * (pos_error**2))
    rot_tracking_rew = -torch.log(1.e-5 + orient_error**2) 
    #rot_tracking_rew = torch.exp(-(1.0/sigma) * (orient_error**2))
    
    return pos_tracking_rew + rot_tracking_rew

def position_command_error_ln(env: RLTaskEnv, epsilon: float, command_name: str,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return -torch.log(epsilon + torch.norm(curr_pos_w - des_pos_w, dim=1)**2)

def position_command_error_exp(env: RLTaskEnv, sigma: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm and exponential .

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    # set sigma
    factor = 1.0 / sigma
    
    return torch.exp(-factor * torch.norm(curr_pos_w - des_pos_w, dim=1))

def orientation_command_error_ln(env: RLTaskEnv, epsilon: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return -torch.log(epsilon + quat_error_magnitude(curr_quat_w, des_quat_w))

def orientation_command_error_exp(env: RLTaskEnv, sigma: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    # set sigma
    factor = 1.0 / sigma

    return torch.exp(-factor * quat_error_magnitude(curr_quat_w, des_quat_w))


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

    return (1.0 - 0.5 * torch.abs(curr_heading_w - des_heading_w)) #* (env.command_manager.get_term(command_name).time_left < 1.0)

def move_in_direction_reward(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Calculates the reward for moving in the direction of the command in a reinforcement learning task environment.

    Args:
        env (RLTaskEnv): The reinforcement learning task environment.
        command_name (str): The name of the command.
        asset_cfg (SceneEntityCfg): The configuration of the scene entity.

    Returns:
        torch.Tensor: The reward for moving in the direction of the command.

    """
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name) # Assuming 2D Pose Command, shape is (num_envs, 4), where 4 is (x, y, z, heading)
    # obtain the desired heading direction
    target_vec = (command[:, :3] + env.scene.env_origins[:, :3]) - asset.data.root_state_w[:, :3]  # type: ignore
    target_vec = target_vec / torch.norm(target_vec, dim=1, keepdim=True)
    # compute the current heading direction
    curr_vel_direction = asset.data.root_lin_vel_w[:, :3]  # type: ignore
    # compute the dot product between the current and desired heading directions
    return torch.cosine_similarity(curr_vel_direction, target_vec, dim=1)

def dont_wait(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min_vel: float = 0.1) -> torch.Tensor:
    """
    Calculates the reward for not waiting in a reinforcement learning task environment.

    Args:
        env (RLTaskEnv): The reinforcement learning task environment.
        command_name (str): The name of the command.

    Returns:
        torch.Tensor: The reward for not waiting.

    """
    asset: Articulation = env.scene[asset_cfg.name]

    return torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1) < min_vel

def stand_at_target(env: RLTaskEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
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
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.root_pos_w

    des_heading_w = command[:, 3]
    # compute the current heading direction
    curr_heading_w = torch.atan2(asset.data.root_quat_w[:, 1], asset.data.root_quat_w[:, 0])

    # Check if target has been reached
    position_reached = torch.norm(curr_pos_w[:, :2] - des_pos_w[:, :2], dim=1) < 0.1
    heading_reached = torch.abs(curr_heading_w - des_heading_w) < 0.1

    # Compute difference between desired and default joint positions
    joint_deviation = torch.norm(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids], dim=1)

    return position_reached * heading_reached * joint_deviation

