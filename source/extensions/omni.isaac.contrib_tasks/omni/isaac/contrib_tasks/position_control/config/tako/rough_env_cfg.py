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
