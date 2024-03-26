from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv, BaseEnv

def apply_feet_adhesion_force(
    env: BaseEnv,
    env_ids: torch.Tensor,
    adhesion_force: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] #type: ignore
    num_envs = env.scene.num_envs
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(num_envs)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    
    # Compute the number of feet contacts
    net_contact_forces = contact_sensor.data.net_forces_w
    feet_contacts = net_contact_forces[:, sensor_cfg.body_ids, 2] > 1.0
    contact_indices = feet_contacts.nonzero(as_tuple=False)

    # create the forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = torch.zeros(size, device=asset.device)
    torques = torch.zeros(size, device=asset.device)
    if contact_indices.numel() == 0:
        asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)
        return
    else:
        forces[:, contact_indices, 2] = -adhesion_force
        # set the forces and torques into the buffers
        # note: these are only applied when you call: `asset.write_data_to_sim()`
        asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)
