from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from collections.abc import Sequence
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv, BaseEnv

def base_position_command_range(
    env: RLTaskEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term("base_pose")
    if env.common_step_counter  % 100 == 0:
        # extract the used quantities (to enable type-hinting)
        # retrieve current metrics
        current_err_mean = torch.mean(command.metrics["error_pos_3d"])
        # Increase difficulty of target pose if robot is performing well
        if current_err_mean < 0.05:
            command.cfg.ranges.pos_x = (command.cfg.ranges.pos_x[0] - 0.05, command.cfg.ranges.pos_x[1] + 0.05)
            command.cfg.ranges.pos_y = (command.cfg.ranges.pos_y[0] - 0.05, command.cfg.ranges.pos_y[1] + 0.05)
            command.cfg.ranges.pos_z = (command.cfg.ranges.pos_z[0] - 0.05, command.cfg.ranges.pos_z[1] + 0.05)
        
    # return the maximum range of the base position command
    return torch.tensor(command.cfg.ranges.pos_x[1] - command.cfg.ranges.pos_x[0], device=env.device)
