# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.orbit.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import RayCaster
from omni.isaac.orbit.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv, RLTaskEnv

"""
Root state.
"""

"""
Joint state.
"""

"""
Sensors.
"""

def feet_contacts(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculates the number of feet in contact with the ground based on a force threshold in the Z direction.

    Args:
        env (BaseEnv): The environment object.
        sensor_cfg (SceneEntityCfg): The configuration of the sensor.

    Returns:
        torch.Tensor: A tensor representing the number of feet in contact with the ground.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check the number of feet in contact with the ground by using a force threshold in the Z direction
    feet_in_contact = net_contact_forces[:, :, sensor_cfg.body_ids, 2] > 1.0

    return feet_in_contact

"""
Actions.
"""

"""
Commands.
"""
