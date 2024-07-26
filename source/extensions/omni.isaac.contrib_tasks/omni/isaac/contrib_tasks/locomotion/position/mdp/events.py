from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv, BaseEnv

def apply_docking_force(
    env: RLTaskEnv,
    env_ids: torch.Tensor,
    dock_force: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    num_envs = env.scene.num_envs
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(num_envs)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    
    # Get desired docking state
    desired_docking_state = env.obs_buf[142:146]
    # Get current docking positions
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids].view(-1, num_bodies ,3)
    docking_state = desired_docking_state * (curr_pos_w[:, :, 2] < 0.1)

    # create the forces and torques
    forces = torch.zeros((num_envs, num_bodies, 3), device=env.device)
    forces[:, :, 0] = dock_force * docking_state
    torques = torch.zeros_like(forces) # No torques are applied
    asset.set_external_force_and_torque(forces, torques=torques, env_ids=env_ids, body_ids=asset_cfg.body_ids) # type: ignore
