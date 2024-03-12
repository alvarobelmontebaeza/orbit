from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def joint_power_l2(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Calculates the L2 norm of the joint power for a given environment and asset configuration.

    Args:
        env (RLTaskEnv): The RLTaskEnv object representing the environment.
        asset_cfg (SceneEntityCfg, optional): The SceneEntityCfg object representing the asset configuration. Defaults to SceneEntityCfg("robot").

    Returns:
        torch.Tensor: The L2 norm of the joint power.

    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    power = torch.clamp(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids], min=0.0)
    return torch.sum(torch.square(power), dim=1)
