# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# local imports
from source.standalone.workflows.rsl_rl import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--bag_path", type=str, default=None, help="Path to the rosbag file")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)

import rosbag

def read_rosbag(file_path, topic='/xpp/state_des'):
    """
    Reads a ROS bag file and extracts messages from a specified topic.

    Args:
        file_path (str): The path to the ROS bag file.
        topic (str, optional): The topic to extract messages from. Defaults to '/xpp/state_des'.

    Returns:
        list: A list of messages extracted from the specified topic.
    """
    bag = rosbag.Bag(file_path)
    num_msg = bag.get_message_count()
    print(f"Number of messages in all bagfile: {num_msg}")
    msg_buff = []
    for topic, msg, t in bag.read_messages(topics=topic):
        msg_buff.append(msg)
    bag.close()

    print(f"Number of messages extracted: {len(msg_buff)}")
       
    return msg_buff


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=1, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    ############# START EXECUTION LOOP #############

    # reset environment
    obs, _ = env.reset()
    #obs, _ = env.get_observations()

    # Get trajectory
    # Read Rosbag file to extract de trajectory
    rosbag_file = args_cli.bag_path
    if rosbag_file is None:
        raise ValueError("Please provide a path to the rosbag file.")
    traj_msgs = read_rosbag(rosbag_file)
    num_traj_points = len(traj_msgs)
    # Configuration
    leg_id = ["LH", "RF", "LF", "RH"]
    traj_time = 50.0 #s
    dt = env.unwrapped.step_dt
    num_points = int(traj_time / dt)
    tracking_point_update_rate = num_points // num_traj_points
    extra_points = num_points % num_traj_points
    print(f"Number of trajectory points: {tracking_point_update_rate}")
    print(f"Number of points: {num_points}")

    # Sample per-leg initial position and generate trajectories
    planned_base_traj = torch.zeros((1, num_points, 7), device=env.unwrapped.device)
    planned_arm_traj = torch.zeros((4, num_points, 3), device=env.unwrapped.device)
    planned_docking_state = torch.zeros((num_points, 4), device=env.unwrapped.device)
    real_base_traj = torch.zeros((1, num_points, 7))
    real_arm_traj = torch.zeros((4, num_points, 3))

    # Parse trajectory messages
    idx = 0
    for i in range(num_points):        
        # base trajectory
        planned_base_traj[0, i, 0] = traj_msgs[idx].base.pose.position.x
        planned_base_traj[0, i, 1] = traj_msgs[idx].base.pose.position.y
        planned_base_traj[0, i, 2] = traj_msgs[idx].base.pose.position.z
        planned_base_traj[0, i, 3] = traj_msgs[idx].base.pose.orientation.w
        planned_base_traj[0, i, 4] = traj_msgs[idx].base.pose.orientation.x
        planned_base_traj[0, i, 5] = traj_msgs[idx].base.pose.orientation.y
        planned_base_traj[0, i, 6] = traj_msgs[idx].base.pose.orientation.z

        # Arm trajectory
        # Planned arm ee poses are given in world frame, but policy expects them in base frame, so we need to convert them
        for arm in range(4):
            planned_arm_traj[arm, i, 0] = traj_msgs[idx].ee_motion[arm].pos.x - planned_base_traj[0, i, 0]
            planned_arm_traj[arm, i, 1] = traj_msgs[idx].ee_motion[arm].pos.y - planned_base_traj[0, i, 1]
            planned_arm_traj[arm, i, 2] = traj_msgs[idx].ee_motion[arm].pos.z - planned_base_traj[0, i, 2] + 0.03 # plan is done wrt the docking, but reference is given to the wrist

        # Planned contact states
        planned_docking_state[i, :] = torch.tensor(traj_msgs[idx].ee_contact, device=env.unwrapped.device)

        # Update index of planned trajectory
        if i % tracking_point_update_rate == 0 and i != 0 and idx < num_traj_points - 1:
            idx += 1


    # Execute trajectory
    steps = 0
    # Set initial goals
    # Override commands with planned trajectory
    env.unwrapped.command_manager.get_term("base_pose").pos_command_b = torch.tensor(planned_base_traj[0, steps, :3], device=env.unwrapped.device).view(1, 3)
    env.unwrapped.command_manager.get_term("base_pose").rot_command_b = torch.tensor(planned_base_traj[0, steps, 3:], device=env.unwrapped.device).view(1, 4)
    for i in range(4):
        env.unwrapped.command_manager.get_term(leg_id[i] + "_pose").pose_command_b[0, 0:3] = torch.tensor(planned_arm_traj[i, steps, :], device=env.unwrapped.device).view(1, 3)
    
    # Do dummy step to update commands
    obs, _, _, _ = env.step(torch.zeros(env.action_space.sample().shape))
    '''
    obs[:, 142] = planned_docking_state[steps, 2] # LF
    obs[:, 143] = planned_docking_state[steps, 0] # LH
    obs[:, 144] = planned_docking_state[steps, 1] # RF
    obs[:, 145] = planned_docking_state[steps, 3] # RH
    '''

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # Override commands with planned trajectory
            env.unwrapped.command_manager.get_term("base_pose").pos_command_w = torch.tensor(planned_base_traj[0, steps, :3], device=env.unwrapped.device).view(1, 3)
            env.unwrapped.command_manager.get_term("base_pose").rot_command_w = torch.tensor(planned_base_traj[0, steps, 3:], device=env.unwrapped.device).view(1, 4)
            for i in range(4):
                env.unwrapped.command_manager.get_term(leg_id[i] + "_pose").pose_command_b[0, 0:3] = torch.tensor(planned_arm_traj[i, steps, :], device=env.unwrapped.device).view(1, 3)
            '''
            # Update docking state
            obs[:, 142] = planned_docking_state[steps, 2] # LF
            obs[:, 143] = planned_docking_state[steps, 0] # LH
            obs[:, 144] = planned_docking_state[steps, 1] # RF
            obs[:, 145] = planned_docking_state[steps, 3] # RH
            '''

            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            # Update docking state
            '''
            obs[:, 142] = planned_docking_state[steps, 2] # LF
            obs[:, 143] = planned_docking_state[steps, 0] # LH
            obs[:, 144] = planned_docking_state[steps, 1] # RF
            obs[:, 145] = planned_docking_state[steps, 3] # RH
            

            # Get real trajectory
            real_base_traj[0, steps, 0] = obs[0, 0].item() #TODO: Add orientation
            real_base_traj[0, steps, 1] = obs[0, 1].item()
            real_base_traj[0, steps, 2] = obs[0, 2].item()

            real_arm_traj[0, steps, 0] = obs[0, 133].item() # LH
            real_arm_traj[0, steps, 1] = obs[0, 134].item()
            real_arm_traj[0, steps, 2] = obs[0, 135].item()


            real_arm_traj[1, steps, 0] = obs[0, 136].item() # RF
            real_arm_traj[1, steps, 1] = obs[0, 137].item()
            real_arm_traj[1, steps, 2] = obs[0, 138].item()

            real_arm_traj[2, steps, 0] = obs[0, 130].item() # LF
            real_arm_traj[2, steps, 1] = obs[0, 131].item()
            real_arm_traj[2, steps, 2] = obs[0, 132].item()

            real_arm_traj[3, steps, 0] = obs[0, 139].item() # RH
            real_arm_traj[3, steps, 1] = obs[0, 140].item()
            real_arm_traj[3, steps, 2] = obs[0, 141].item()
            '''

            steps += 1
            if steps == num_points:
                # Store last point
                # TODO
                plt.figure(figsize=(6, 6))
                plt.plot(np.arange(num_points), planned_base_traj[0, :, 0].numpy(), 'b-')
                plt.show()
                break

    # close the simulator
    env.close()

def generate_semicircular_trajectory(center, radius, num_points):
    angles = torch.linspace(0, torch.tensor(3.14159265358979323846), num_points)
    x = center[0] - radius * torch.cos(angles)
    y = center[1] + radius * torch.sin(angles)
    return x, y

def generate_linear_trajectory_x_axis(x0, length, num_points):
    t = torch.linspace(0, 1, num_points)
    trajectory = x0 + t.view(-1, 1) * length * torch.tensor([1.0, 1.0])  # Direction along X-axis
    return trajectory[:, 0], trajectory[:, 1]

def plot_circular_trajectory(x_plan, y_plan, x_real, y_real, center, leg_name="LF"):
    # Plotting the circular trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(x_plan.numpy(), y_plan.numpy(), 'b-')
    plt.plot(x_real.numpy(), y_real.numpy(), 'r-')
    plt.plot(center[0].item(), center[1].item(), 'go')  # Plotting the center point
    plt.axis('equal')
    plt.title(leg_name + ' Arc Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.legend(['Planned', 'Real'])
    plt.grid(True)
    plt.show()

def plot_linear_trajectory(x_plan, y_plan, x_real, y_real, leg_name="LF"):
    # Plotting the linear trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(x_plan.numpy(), y_plan.numpy(), 'b-')
    plt.plot(x_real.numpy(), y_real.numpy(), 'r-')
    plt.axis('equal')
    plt.title(leg_name + ' Linear Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.legend(['Planned', 'Real'])
    plt.grid(True)
    plt.show()

def plot_real_vs_expected(x_plan, y_plan, z_plan, x_real, y_real, z_real, sec_per_point=0.05, leg_name="LF"):
    # Plotting the real vs expected trajectory
    time = np.arange(1, len(x_plan)+1) * sec_per_point
    plt.figure(figsize=(6, 6))
    plt.plot(time, x_plan.numpy(), 'm-')
    plt.plot(time, x_real.numpy(), 'r-')
    plt.plot(time, y_plan.numpy(), 'g-')
    plt.plot(time, y_real.numpy(), 'y-')
    plt.plot(time, z_plan.numpy(), 'c-')
    plt.plot(time, z_real.numpy(), 'b-')
    plt.title(leg_name + ' Real vs Planned Trajectory')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend(['X-Plan', 'X-Real', 'Y-Plan', 'Y-Real', 'Z-Plan', 'Z-Real'])
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
