# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.orbit.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.managers import SceneEntityCfg


from omni.isaac.contrib_tasks.leg_position_control.mdp.actions import ee_actions, base_motion_actions

##
# End-effector actions
##


@configclass
class GripForceActionCfg(ActionTermCfg):
    """
    Configuration class for the GripForceAction.

    Attributes:
        class_type (type[ActionTerm]): The type of the action term.
        ee_names (list[str]): List of joint names or regex expressions that the action will be mapped to.
        max_force (float | dict[str, float]): Scale factor for the action (float or dict of regex expressions). Defaults to 1.0.
        threshold (float | dict[str, float]): Offset factor for the action (float or dict of regex expressions). Defaults to 0.0.
        sensor_cfg (SceneEntityCfg): Configuration for the scene entity.
    """

    class_type: type[ActionTerm] = ee_actions.GripForceAction

    ee_names: list[str] = MISSING #type: ignore
    """List of body ids or regex expressions that the action will be mapped to."""

    max_force: float | dict[str, float] = 5.0
    """Maximum force that will be applied to the bodies. Defaults to 5.0."""

    threshold: float | dict[str, float] = 1.0
    """Minimum value of the contact force to be considered a real contact. Defaults to 1.0."""

    sensor_cfg: SceneEntityCfg = SceneEntityCfg()

@configclass
class BodyThrusterActionCfg(ActionTermCfg):
    """
    Configuration class for the GripForceAction.

    Attributes:
        class_type (type[ActionTerm]): The type of the action term.
        ee_names (list[str]): List of joint names or regex expressions that the action will be mapped to.
        max_force (float | dict[str, float]): Scale factor for the action (float or dict of regex expressions). Defaults to 1.0.
        threshold (float | dict[str, float]): Offset factor for the action (float or dict of regex expressions). Defaults to 0.0.
        sensor_cfg (SceneEntityCfg): Configuration for the scene entity.
    """

    class_type: type[ActionTerm] = base_motion_actions.BodyThrusterAction

    max_push_force: float | dict[str, float] = 5.0
    """Maximum force that will be applied to the bodies. Defaults to 5.0."""

    threshold: float | dict[str, float] = 1.0
    """Minimum value of the contact force to be considered a real contact. Defaults to 1.0."""





