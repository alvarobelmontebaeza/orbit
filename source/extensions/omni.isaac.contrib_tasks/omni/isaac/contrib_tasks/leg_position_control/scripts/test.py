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
    obs, _, _, _ = env.step(torch.zeros(env.action_space.sample().shape))
    #obs, _ = env.get_observations()

    # Get trajectory
    # Configuration
    traj_time = 10.0 #s
    dt = env.unwrapped.step_dt
    tracking_point_update_rate = 10 #steps per update
    num_points = int((traj_time / dt)/ tracking_point_update_rate) 
    print(f"Number of points: {num_points}")

    # Sample per-leg initial position and generate trajectories
    radius = 0.2
    init_pos = torch.zeros((4, 3))
    planned_traj = torch.zeros((4, num_points, 3))
    real_traj = torch.zeros((4, num_points, 3))
    for i in range(4):
        offset = i * 3     
        init_pos[i,0], init_pos[i,1], init_pos[i,2] = obs[0, 84+offset].item(), obs[0, 85+offset].item(), obs[0,86+offset].item() # initial position

        # Generate circular trajectory
        x_plan, z_plan = generate_circular_trajectory(center=torch.tensor([init_pos[i,0]-radius, init_pos[i,2]]), radius=radius, num_points=num_points+1)
        planned_traj[i, :, 0] = x_plan[1:]
        planned_traj[i, :, 1] = torch.ones(num_points) * init_pos[i,1]
        planned_traj[i, :, 2] = z_plan[1:]

    # Execute trajectory
    idx = 0
    steps = 0

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # Override position command for each leg
            # LF
            obs[:,56:59] = torch.tensor([planned_traj[0, idx, 0], planned_traj[0, idx, 1], planned_traj[0, idx, 2]], device=obs.device)
            # LH
            obs[:,63:66] = torch.tensor([planned_traj[1, idx, 0], planned_traj[1, idx, 1], planned_traj[1, idx, 2]], device=obs.device)
            # RF
            obs[:,70:73] = torch.tensor([planned_traj[2, idx, 0], planned_traj[2, idx, 1], planned_traj[2, idx, 2]], device=obs.device)
            # RH
            obs[:,77:80] = torch.tensor([planned_traj[3, idx, 0], planned_traj[3, idx, 1], planned_traj[3, idx, 2]], device=obs.device)

            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            if idx < (len(x_plan) - 1):
                # Store real trajectory
                if steps % tracking_point_update_rate == 0:
                    for i in range(4):
                        offset = i * 3
                        real_traj[i, idx, 0] = obs[0, 84+offset].item()
                        real_traj[i, idx, 1] = obs[0, 85+offset].item()
                        real_traj[i, idx, 2] = obs[0, 86+offset].item()
                    idx += 1 

            steps = env.unwrapped.common_step_counter
               
            if steps == int(traj_time / dt) + tracking_point_update_rate:
                # Store last point
                for i in range(4):
                    offset = i * 3
                    real_traj[i, idx, 0] = obs[0, 84+offset].item()
                    real_traj[i, idx, 1] = obs[0, 85+offset].item()
                    real_traj[i, idx, 2] = obs[0, 86+offset].item()
                
                print(f"Mean tracking error: {torch.mean(torch.abs(planned_traj - real_traj))}")
                # Plot trajectories for each leg
                for i in range(4):
                    print(f"Leg {i}: Circular trajectory")
                    plot_circular_trajectory(planned_traj[i, :, 0], planned_traj[i, :, 2], real_traj[i, :, 0], real_traj[i, :, 2], torch.tensor([init_pos[i,0], init_pos[i,2]]), title=f"Leg {i}: Circular Trajectory")
                    #plot_linear_trajectory(x_plan, z_plan, x_real, z_real)
                    plot_real_vs_expected(planned_traj[i, :, 0], planned_traj[i, :, 2], real_traj[i, :, 0], real_traj[i, :, 2], sec_per_point=dt*tracking_point_update_rate)
                break

    # close the simulator
    env.close()

def generate_circular_trajectory(center, radius, num_points):
    angles = torch.linspace(0, 2 * torch.tensor(3.14159265358979323846), num_points)
    x = center[0] + radius * torch.cos(angles)
    y = center[1] + radius * torch.sin(angles)
    return x, y

def generate_linear_trajectory_x_axis(x0, length, num_points):
    t = torch.linspace(0, 1, num_points)
    trajectory = x0 + t.view(-1, 1) * length * torch.tensor([1.0, 0.0])  # Direction along X-axis
    return trajectory[:, 0], trajectory[:, 1]

def plot_circular_trajectory(x_plan, y_plan, x_real, y_real, center, title="Circular Trajectory"):
    # Plotting the circular trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(x_plan.numpy(), y_plan.numpy(), 'b-')
    plt.plot(x_real.numpy(), y_real.numpy(), 'r-')
    plt.plot(center[0].item(), center[1].item(), 'go')  # Plotting the center point
    plt.axis('equal')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.grid(True)
    plt.show()

def plot_linear_trajectory(x_plan, y_plan, x_real, y_real):
    # Plotting the linear trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(x_plan.numpy(), y_plan.numpy(), 'b-')
    plt.plot(x_real.numpy(), y_real.numpy(), 'r-')
    plt.axis('equal')
    plt.title('Linear Trajectory')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.grid(True)
    plt.show()

def plot_real_vs_expected(x_plan, y_plan, x_real, y_real, sec_per_point=0.05):
    # Plotting the real vs expected trajectory
    time = np.arange(1, len(x_plan)+1) * sec_per_point
    plt.figure(figsize=(6, 6))
    plt.plot(time, x_plan.numpy(), 'b-')
    plt.plot(time, x_real.numpy(), 'r-')
    plt.plot(time, y_plan.numpy(), 'g-')
    plt.plot(time, y_real.numpy(), 'y-')
    plt.title('Real vs Planned Trajectory')
    plt.xlabel('Time')
    plt.ylabel('Position (m)')
    plt.legend(['X-Plan', 'X-Real', 'Z-Plan', 'Z-Real'])
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
