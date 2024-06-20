# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.utils import configclass
import math

from omni.isaac.contrib_tasks.locomotion.position.position_env_cfg import LocomotionPositionRoughEnvCfg
import omni.isaac.contrib_tasks.locomotion.position.mdp as mdp
##
# Pre-defined configs
##
from omni.isaac.contrib_assets.tako import TAKO_CFG  # isort: skip


@configclass
class TakoRoughEnvCfg(LocomotionPositionRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to Tako
        self.scene.robot = TAKO_CFG.replace(prim_path="{ENV_REGEX_NS}/tako")

        feet_init_pos = {
            "LF": [0.55, 0.9, -0.5],
            "LH": [-0.3, 0.9, -0.5],
            "RF": [0.55, -0.73, -0.5],
            "RH": [-0.3, -0.73, -0.5],
        }
        # override command generator
        # Set command generator to sample points around the foot initial position
        for leg_prefix in feet_init_pos.keys():
            command_name = leg_prefix + "_pose"
            leg_command = getattr(self.commands, command_name)
            leg_command.ranges.pos_x = (feet_init_pos[leg_prefix][0] - 0.2, feet_init_pos[leg_prefix][0] + 0.2)
            leg_command.ranges.pos_y = (feet_init_pos[leg_prefix][1] - 0.2, feet_init_pos[leg_prefix][1] + 0.2)
            leg_command.ranges.pos_z = (feet_init_pos[leg_prefix][2] - 0.2, feet_init_pos[leg_prefix][2] + 0.2)
        
        # Set base position command height ranges
        self.commands.base_position.ranges.pos_z = (0.8*self.scene.robot.init_state.pos[2], 1.2*self.scene.robot.init_state.pos[2])


@configclass
class TakoRoughEnvCfg_PLAY(TakoRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None
