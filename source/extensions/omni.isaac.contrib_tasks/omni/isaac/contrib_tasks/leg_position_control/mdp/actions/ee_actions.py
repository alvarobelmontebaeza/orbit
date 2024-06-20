# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb

import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.managers.action_manager import ActionTerm

from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.managers import SceneEntityCfg



if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

    from . import actions_cfg


class GripForceAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.GripForceActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _max_force: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _threshold: torch.Tensor | float
    """The threshold applied to the input action."""
    _sensor_cfg: SceneEntityCfg

    def __init__(self, cfg: actions_cfg.GripForceActionCfg, env: BaseEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the bodies over which the action term is applied
        self._ee_bodies, _ = self._asset.find_bodies(self.cfg.ee_names)
        self._num_bodies = len(self._ee_bodies)
        # log the resolved joint names for debugging
        carb.log_info(
            f"Resolved body names for the action term {self.__class__.__name__}:"
            f" {self._ee_bodies}"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_bodies == self._asset.num_bodies:
            self._ee_bodies = slice(None)
        
        # Instantiate contact sensor for the bodies        
        self._sensor_cfg = cfg.sensor_cfg
        self._contact_sensor: ContactSensor = env.scene.sensors[self._sensor_cfg.name] # type: ignore

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse max_force
        if isinstance(cfg.max_force, (float, int)):
            self._max_force = float(cfg.max_force)
        else:
            raise ValueError(f"Unsupported max_force type: {type(cfg.max_force)}. Supported types are float")
        # parse threshold
        if isinstance(cfg.threshold, (float, int)):
            self._threshold = float(cfg.threshold)
        else:
            raise ValueError(f"Unsupported threshold type: {type(cfg.threshold)}. Supported types are float")
        

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_bodies

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions
        # Check if the contact sensor is activated
        air_time = self._contact_sensor.data.current_air_time[:, self._ee_bodies]
        # Check conatcts in Z axis
        no_contact = air_time > 0.0
        # Make the action zero if the contact sensor is not activated
        if torch.any(no_contact):
            self._processed_actions[no_contact] = 0.0
        
        # Scale the action by the max force
        self._processed_actions *= self._max_force

        # Clip the action to not exceed the max force
         
        self._processed_actions = torch.clip(
            input=self._processed_actions,
            min=0.0,
            max=self._max_force,
        ) # type: ignore

    def apply_actions(self):
        # apply forces to each body
        env_ids = torch.arange(self.num_envs, device=self.device)
        #self._asset.body_physx_view.set_velocities(torch.zeros((self.num_envs, self._ee_bodies[0], 6), indices=self._ee_bodies[0],device=self.device)) # type: ignore
        grip_forces = torch.zeros((self.num_envs, self._num_bodies, 3), device=self.device)
        grip_forces[:, :, 2] = self._processed_actions
        torques = torch.zeros_like(grip_forces) # No torques are applied
        self._asset.set_external_force_and_torque(grip_forces, torques=torques, env_ids=env_ids, body_ids=self._ee_bodies) # type: ignore



