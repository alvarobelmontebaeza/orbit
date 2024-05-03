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
    traj_time = 15.0 #s
    dt = env.unwrapped.step_dt
    tracking_point_update_rate = 10 #steps per update
    num_points = int((traj_time / dt)/ tracking_point_update_rate) 
    print(f"Number of points: {num_points}")
    x0, y0, z0 = obs[0, -9].item(), obs[0, -8].item(), obs[0, -7].item() # initial position
    radius = 0.2
    print(f"Initial position: ({x0}, {y0}, {z0})")

    x_plan, z_plan = generate_circular_trajectory(center=torch.tensor([x0-radius, z0]), radius=radius, num_points=num_points+1)
    #x_plan, z_plan = generate_linear_trajectory_x_axis(torch.tensor([x0, z0]), length=0.2, num_points=num_points+1)
    # Delete first point since it is the initial position
    x_plan = x_plan[1:]
    z_plan = z_plan[1:]
    x_real = torch.zeros_like(x_plan)
    z_real = torch.zeros_like(z_plan)
    idx = 0
    steps = 0

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # Override position command
            obs[:,-16:-13] = torch.tensor([x_plan[idx], y0, z_plan[idx]], device=obs.device)
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            if idx < (len(x_plan) - 1):
                # Store real trajectory
                if steps % tracking_point_update_rate == 0:
                    x_real[idx] = obs[0, -9]
                    z_real[idx] = obs[0, -7]
                    idx += 1 

            steps = env.unwrapped.common_step_counter
               
            if steps == int(traj_time / dt) + tracking_point_update_rate:
                # Store last point
                x_real[idx] = obs[0, -9]
                z_real[idx] = obs[0, -7]
                
                print(f"Mean tracking error: {torch.mean(torch.abs(x_plan - x_real))}" )
                plot_circular_trajectory(x_plan, z_plan, x_real, z_real, torch.tensor([x0, z0]))
                #plot_linear_trajectory(x_plan, z_plan, x_real, z_real)
                plot_real_vs_expected(x_plan, z_plan, x_real, z_real, sec_per_point=dt*tracking_point_update_rate)
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

def plot_circular_trajectory(x_plan, y_plan, x_real, y_real, center):
    # Plotting the circular trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(x_plan.numpy(), y_plan.numpy(), 'b-')
    plt.plot(x_real.numpy(), y_real.numpy(), 'r-')
    plt.plot(center[0].item(), center[1].item(), 'go')  # Plotting the center point
    plt.axis('equal')
    plt.title('Circular Trajectory')
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
