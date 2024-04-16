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

        # switch robot to ur10
        self.scene.robot = TAKO_CFG
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.ee_pos_tracking.params["asset_cfg"].body_names = [foot_name]
        self.rewards.ee_orient_tracking.params["asset_cfg"].body_names = [foot_name]
        # override actions
        self.actions.leg_action = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=[leg_prefix + ".*"], scale=0.5
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = foot_name
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class TakoLegPositionControlEnvCfg_PLAY(LegPositionControlEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
