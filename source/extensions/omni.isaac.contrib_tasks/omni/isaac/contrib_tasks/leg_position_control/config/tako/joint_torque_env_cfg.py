# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

from omni.isaac.orbit.utils import configclass

import omni.isaac.contrib_tasks.leg_position_control.mdp as mdp
from omni.isaac.contrib_tasks.leg_position_control.leg_position_control_env_cfg import LegPositionControlEnvCfg

##
# Pre-defined configs
##
from omni.isaac.contrib_assets import TAKO_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class TakoLegPositionControlEnvCfg(LegPositionControlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        leg_prefix = "LF"
        foot_name = leg_prefix + "_gecko"
        # Feet init pos
        feet_init_pos = {
            "LF": [0.55, 0.9, -0.5],
            "LH": [-0.3, 0.9, -0.5],
            "RF": [0.55, -0.73, -0.5],
            "RH": [-0.3, -0.73, -0.5],
        }

        self.scene.robot = TAKO_CFG.replace(prim_path="{ENV_REGEX_NS}/tako")
        self.scene.robot.usd_path = "/home/alvaro/Desktop/tako_leg_pos_control.usd"
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # For now, remove randomization events
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # override rewards
        self.rewards.ee_pos_tracking.params["asset_cfg"].body_names = [foot_name]
        self.rewards.ee_orient_tracking.params["asset_cfg"].body_names = [foot_name]
        # override actions
        '''
        self.actions.leg_action = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=[leg_prefix + "_shoulder_pan_joint", leg_prefix + "_shoulder_lift_joint", leg_prefix + "_elbow_joint"], scale=150.0, 
        )
        self.actions.foot_action = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=[leg_prefix + "_wrist_1_joint", leg_prefix + "_wrist_2_joint", leg_prefix + "_wrist_3_joint"], scale=28.0, 
        )
        '''
        self.actions.leg_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[leg_prefix + ".*"], scale=0.5
        )
        # Override observations
        
        self.observations.policy.foot_pos.params["asset_cfg"].body_names = [foot_name]
        self.observations.policy.foot_orient.params["asset_cfg"].body_names = [foot_name]
        
        
        # override command generator
        # end-effector is along Z-direction
        self.commands.ee_pose.body_name = foot_name

        # Set command generator to sample points around the foot initial position
        self.commands.ee_pose.ranges.pos_x = (feet_init_pos[leg_prefix][0] - 0.4, feet_init_pos[leg_prefix][0] + 0.4)
        self.commands.ee_pose.ranges.pos_y = (feet_init_pos[leg_prefix][1] - 0.3, feet_init_pos[leg_prefix][1] + 0.3)
        self.commands.ee_pose.ranges.pos_z = (feet_init_pos[leg_prefix][2] - 0.4, feet_init_pos[leg_prefix][2] + 0.4)

        # Set commanded orientation to face the ground +- an error
        self.commands.ee_pose.ranges.roll = (math.pi, math.pi)
        self.commands.ee_pose.ranges.pitch = (0.0, 0.0)
        self.commands.ee_pose.ranges.yaw = (0.0, 0.0)




@configclass
class TakoLegPositionControlEnvCfg_PLAY(TakoLegPositionControlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
