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
            "LF": [0.553, 0.924, -0.496],
            "LH": [-0.297, 0.929, -0.483],
            "RF": [0.547, -0.731, -0.495],
            "RH": [-0.295, -0.718, -0.486],
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
        # Remove rewards for non-used legs
        self.rewards.LH_pos_tracking = None
        self.rewards.LH_orient_tracking = None
        self.rewards.RF_pos_tracking = None
        self.rewards.RF_orient_tracking = None
        self.rewards.RH_pos_tracking = None
        self.rewards.RH_orient_tracking = None

        # Adjust reward weights
        self.rewards.dof_power_l2.weight = -0.05
        self.rewards.action_rate_l2.weight = -0.01

        # Remove observations for non-used legs
        self.observations.policy.LH_foot_pos_des = None
        self.observations.policy.RF_foot_pos_des = None
        self.observations.policy.RH_foot_pos_des = None

        self.observations.policy.LH_foot_pos = None
        self.observations.policy.RF_foot_pos = None
        self.observations.policy.RH_foot_pos = None

        # Remove commands for non-used legs
        self.commands.LH_pose = None
        self.commands.RF_pose = None
        self.commands.RH_pose = None

        # override command generator
        # Set command generator to sample points around the foot initial position
        self.commands.LF_pose.ranges.pos_x = (feet_init_pos[leg_prefix][0] - 0.2, feet_init_pos[leg_prefix][0] + 0.2)
        self.commands.LF_pose.ranges.pos_y = (feet_init_pos[leg_prefix][1] - 0.2, feet_init_pos[leg_prefix][1] + 0.2)
        self.commands.LF_pose.ranges.pos_z = (feet_init_pos[leg_prefix][2] - 0.2, feet_init_pos[leg_prefix][2] + 0.2)

        # override actions
        self.actions.legs_joint_position = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=[leg_prefix + ".*"], scale=1.0,
        )
        
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

