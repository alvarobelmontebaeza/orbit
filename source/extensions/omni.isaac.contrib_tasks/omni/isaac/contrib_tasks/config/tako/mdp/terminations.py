
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

"""
Tako MDP terminations.
"""

"""
Contact sensor.
"""


def illegal_contact(env: RLTaskEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )

def feet_contact_num(env: RLTaskEnv, threshold: int, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the number of feet in contact with the ground is less than the threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if the number of feet in contact with the ground is less than the threshold
    feet_in_contact = torch.sum(
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_names] > 0.0, dim=1
    )
    return feet_in_contact < threshold