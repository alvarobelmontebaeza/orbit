# Copyright (c) 2024, Alvaro Belmonte Baeza
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Tako multipod robot 


"""

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg

##
# Configuration
##
TAKO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/alvaro/NVOmniverse/Assets/tako_backup.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*shoulder_pan_joint": 0.0, # all HIP joints
            ".*shoulder_lift_joint": 0.0, # all shoulder joints
            "LF_elbow_joint": 0.0, # left front elbow
            "LH_elbow_joint": 0.0, # left hind elbow
            "RF_elbow_joint": 0.0, # right front elbow
            "RH_elbow_joint": 0.0, # right hind elbow
            ".*wrist_1_joint": 0.0, # all wrist 1 joints
            ".*wrist_2_joint": 0.0, # all wrist 2 joints
            ".*wrist_3_joint": 0.0, # all wrist 3 joints
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(   
            joint_names_expr=[".*shoulder_pan_joint", ".*shoulder_lift_joint", ".*elbow_joint"],
            velocity_limit=None,
            effort_limit=150.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*wrist_.*_joint"],
            velocity_limit=None,
            effort_limit=28.0,
            stiffness=0.0,
            damping=0.0,
        )
    },
    soft_joint_pos_limit_factor=0.95, # Limit factor for joint position limits
)


