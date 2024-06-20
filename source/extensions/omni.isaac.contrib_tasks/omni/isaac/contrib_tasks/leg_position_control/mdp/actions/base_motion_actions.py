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


if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

    from . import actions_cfg


class BodyThrusterAction(ActionTerm):
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

    cfg: actions_cfg.BodyThrusterActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _max_push_force: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _threshold: torch.Tensor | float
    """The threshold applied to the input action."""

    def __init__(self, cfg: actions_cfg.BodyThrusterActionCfg, env: BaseEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        
        # One thruster per cartesian axis
        self._num_thusters = 1

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(size=(self.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions, device=self.device)

        # parse max_force
        if isinstance(cfg.max_push_force, (float, int)):
            self._max_push_force = float(cfg.max_push_force)
        else:
            raise ValueError(f"Unsupported max_force type: {type(cfg.max_push_force)}. Supported types are float")
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
        return self._num_thusters

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
        # Scale the action by the max force
        self._processed_actions = torch.tanh(self._raw_actions) * self._max_push_force # Scale the actions to [-max_push_force, max_push_force]
        self._processed_actions = torch.clip(self._raw_actions, min=-self._max_push_force, max=self._max_push_force)

    def apply_actions(self):
        # apply forces to each body
        env_ids = torch.arange(self.num_envs, device=self.device)
        forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        forces[:, 0, 2] = self._processed_actions.reshape(-1)
        torques = torch.zeros_like(forces) # No torques are applied
        # Find bodies to apply the force
        body_ids, body_names = self._asset.find_bodies(".*body")

        self._asset.set_external_force_and_torque(forces=forces, torques=torques, env_ids=env_ids, body_ids=body_ids) # type: ignore

'''
class BodyThrusterAction(ActionTerm):
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

    cfg: actions_cfg.BodyThrusterActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _max_push_force: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _threshold: torch.Tensor | float
    """The threshold applied to the input action."""

    def __init__(self, cfg: actions_cfg.BodyThrusterActionCfg, env: BaseEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)
        
        # One thruster per cartesian axis
        self._num_thusters = 3

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse max_force
        if isinstance(cfg.max_push_force, (float, int)):
            self._max_push_force = float(cfg.max_push_force)
        else:
            raise ValueError(f"Unsupported max_force type: {type(cfg.max_push_force)}. Supported types are float")
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
        return self._num_thusters

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
        # Scale the action by the max force
        self._processed_actions = torch.tanh(self._raw_actions) * self._max_push_force # Scale the actions to [-max_push_force, max_push_force]
        torch.clamp(self._raw_actions, -self._max_push_force, self._max_push_force, out=self._processed_actions)

    def apply_actions(self):
        # apply forces to each body
        env_ids = torch.arange(self.num_envs, device=self.device)
        forces = torch.zeros(self.num_envs, 1, self._num_thusters, device=self.device)
        forces[:, 0, :] = self._processed_actions
        torques = torch.zeros_like(forces) # No torques are applied
        # Find bodies to apply the force
        body_ids, body_names = self._asset.find_bodies(".*body")

        self._asset.set_external_force_and_torque(forces=forces, torques=torques, env_ids=env_ids, body_ids=body_ids) # type: ignore
'''

