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
        # override actions
        '''        
        self.actions.legs_joint_position = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5,
        )
        '''
        self.actions.legs_joint_position = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1.0,
        )        
        
        # override command generator
        # Set command generator to sample points around the foot initial position
        for leg_prefix in feet_init_pos.keys():
            command_name = leg_prefix + "_pose"
            leg_command = getattr(self.commands, command_name)
            leg_command.ranges.pos_x = (feet_init_pos[leg_prefix][0] - 0.2, feet_init_pos[leg_prefix][0] + 0.2)
            leg_command.ranges.pos_y = (feet_init_pos[leg_prefix][1] - 0.2, feet_init_pos[leg_prefix][1] + 0.2)
            leg_command.ranges.pos_z = (feet_init_pos[leg_prefix][2] - 0.2, feet_init_pos[leg_prefix][2] + 0.2)


class TakoOneLegPositionControlEnvCfg(LegPositionControlEnvCfg):
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

        # Remove parts of the MDP specific for the other legs
        for leg in feet_init_pos.keys():
            if leg_prefix == leg:
                continue
            else:
                # Remove rewards
                pos_reward = getattr(self.rewards, leg + "_pos_tracking")
                orient_reward = getattr(self.rewards, leg + "_orient_tracking")
                pos_reward = None
                orient_reward = None

                # Remove observations
                foot_pos_des_obs = getattr(self.observations.policy, leg + "_foot_pos_des")

                # Remove commands
                leg_command = getattr(self.commands, leg + "_pose")
                leg_command = None                

        # override actions
        self.actions.legs_joint_position = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=[leg_prefix + ".*"], scale=1.0,
        )
        
        # Override observations        
        self.observations.policy.foot_pos.params["asset_cfg"].body_names = [foot_name]
        self.observations.policy.foot_orient.params["asset_cfg"].body_names = [foot_name]

@configclass
class TakoLegPositionControlEnvCfg_PLAY(TakoLegPositionControlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.events.reset_robot_joints = None
        self.events.reset_base = None
        self.observations.policy.enable_corruption = False

@configclass
class TakoOneLegPositionControlEnvCfg_PLAY(TakoOneLegPositionControlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.events.reset_robot_joints = None
        self.events.reset_base = None
        self.observations.policy.enable_corruption = False

