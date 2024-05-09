# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import ActionTermCfg as ActionTerm
from omni.isaac.orbit.managers import EventTermCfg as EventTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

#import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp
import omni.isaac.contrib_tasks.leg_position_control.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.orbit.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from omni.isaac.contrib_assets.tako import TAKO_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a free-floating robot."""

    # ground terrain
    '''
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    '''
    # robots
    robot: ArticulationCfg = TAKO_CFG
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/tako/.*", history_length=3, track_air_time=True)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    LF_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=".*LF_gecko",
        resampling_time_range=(6.0, 6.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(math.pi, math.pi),
            pitch=(0.0, 0.0),  # depends on end-effector axis
            yaw=(0.0, 0.0),
        ),
    )
    LH_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=".*LH_gecko",
        resampling_time_range=(6.0, 6.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(math.pi, math.pi),
            pitch=(0.0, 0.0),  # depends on end-effector axis
            yaw=(0.0, 0.0),
        ),
    )
    RF_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=".*RF_gecko",
        resampling_time_range=(6.0, 6.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(math.pi, math.pi),
            pitch=(0.0, 0.0),  # depends on end-effector axis
            yaw=(0.0, 0.0),
        ),
    )    
    RH_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=".*RH_gecko",
        resampling_time_range=(6.0, 6.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(math.pi, math.pi),
            pitch=(0.0, 0.0),  # depends on end-effector axis
            yaw=(0.0, 0.0),
        ),
    )
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    legs_joint_position: ActionTerm = MISSING

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) # 0-2
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel) # 3-5
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        ) # 6-8
        joint_pos = ObsTerm(func=mdp.joint_pos_rel) # 9 - 32
        joint_vel = ObsTerm(func=mdp.joint_vel_rel) # 32 - 55
        LF_foot_pos_des = ObsTerm(func=mdp.generated_commands, params={"command_name": "LF_pose"}) # 56 - 62
        LH_foot_pos_des = ObsTerm(func=mdp.generated_commands, params={"command_name": "LH_pose"}) # 63 - 69
        RF_foot_pos_des = ObsTerm(func=mdp.generated_commands, params={"command_name": "RF_pose"}) # 70 - 76
        RH_foot_pos_des = ObsTerm(func=mdp.generated_commands, params={"command_name": "RH_pose"}) # 77 - 83
        LF_foot_pos = ObsTerm(func=mdp.foot_position, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*LF_gecko")}) # 84 - 86
        LH_foot_pos = ObsTerm(func=mdp.foot_position, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*LH_gecko")}) # 87 - 89
        RF_foot_pos = ObsTerm(func=mdp.foot_position, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*RF_gecko")}) # 90 - 92
        RH_foot_pos = ObsTerm(func=mdp.foot_position, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*RH_gecko")}) # 93 - 95
        actions = ObsTerm(func=mdp.last_action) # 96 - 120

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for Event."""

    # startup
    '''
    add_base_mass = EventTerm(
        func=mdp.add_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="body"), "mass_range": (-5.0, 5.0)},
    )
    '''

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.), "y": (-0., 0.), "yaw": (-0., 0.)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
    )

'''
# EXPONENTIAL REWARD    
# POSITION TRACKING
weight_pos_track = 15.0
sigma_pos_track = 0.1
# ORIENTATION TRACKING
weight_orient_track = 2.5
sigma_orient_track = 0.5

'''
# LOGARITHMIC REWARD
# POSITION TRACKING
weight_pos_track = 15.0
epsilon_pos_track = 1e-5
# ORIENTATION TRACKING
weight_orient_track = 15.0
epsilon_orient_track = 1e-5




@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task rewards
    '''
    ############# EXPONENTIAL REWARD #############
    LF_pos_tracking = RewTerm(
        func=mdp.position_command_error_exp,
        weight=weight_pos_track,
        params={"sigma": sigma_pos_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*LF_gecko"), "command_name": "LF_pose"},
    )
    LH_pos_tracking = RewTerm(
        func=mdp.position_command_error_exp,
        weight=weight_pos_track,
        params={"sigma": sigma_pos_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*LH_gecko"), "command_name": "LH_pose"},
    ) 
    RF_pos_tracking = RewTerm(
        func=mdp.position_command_error_exp,
        weight=weight_pos_track,
        params={"sigma": sigma_pos_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*RF_gecko"), "command_name": "RF_pose"},
    ) 
    RH_pos_tracking = RewTerm(
        func=mdp.position_command_error_exp,
        weight=weight_pos_track,
        params={"sigma": sigma_pos_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*RH_gecko"), "command_name": "RH_pose"},
    )  

    LF_orient_tracking = RewTerm(
        func=mdp.orientation_command_error_exp,
        weight=weight_orient_track,
        params={"sigma": sigma_orient_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*LF_gecko"), "command_name": "LF_pose"},
    )
    LH_orient_tracking = RewTerm(
        func=mdp.orientation_command_error_exp,
        weight=weight_orient_track,
        params={"sigma": sigma_orient_track,"asset_cfg": SceneEntityCfg("robot", body_names=".*LH_gecko"), "command_name": "LH_pose"},
    )
    RF_orient_tracking = RewTerm(
        func=mdp.orientation_command_error_exp,
        weight=weight_orient_track,
        params={"sigma": sigma_orient_track,"asset_cfg": SceneEntityCfg("robot", body_names=".*RF_gecko"), "command_name": "RF_pose"},
    )
    RH_orient_tracking = RewTerm(
        func=mdp.orientation_command_error_exp,
        weight=weight_orient_track,
        params={"sigma": sigma_orient_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*RH_gecko"), "command_name": "RH_pose"},
    )
    '''
    ############# LOGARITHMIC REWARD #############
    # POSITION TRACKING

    LF_pos_tracking = RewTerm(
        func=mdp.position_command_error_ln,
        weight=weight_pos_track,
        params={"epsilon": epsilon_pos_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*LF_gecko"), "command_name": "LF_pose"},
    )
    LH_pos_tracking = RewTerm(
        func=mdp.position_command_error_ln,
        weight=weight_pos_track,
        params={"epsilon": epsilon_pos_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*LH_gecko"), "command_name": "LH_pose"},
    ) 
    RF_pos_tracking = RewTerm(
        func=mdp.position_command_error_ln,
        weight=weight_pos_track,
        params={"epsilon": epsilon_pos_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*RF_gecko"), "command_name": "RF_pose"},
    ) 
    RH_pos_tracking = RewTerm(
        func=mdp.position_command_error_ln,
        weight=weight_pos_track,
        params={"epsilon": epsilon_pos_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*RH_gecko"), "command_name": "RH_pose"},
    )  

    # ORIENTATION TRACKING

    LF_orient_tracking = RewTerm(
        func=mdp.orientation_command_error_ln,
        weight=weight_orient_track,
        params={"epsilon": epsilon_orient_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*LF_gecko"), "command_name": "LF_pose"},
    )
    LH_orient_tracking = RewTerm(
        func=mdp.orientation_command_error_ln,
        weight=weight_orient_track,
        params={"epsilon": epsilon_orient_track,"asset_cfg": SceneEntityCfg("robot", body_names=".*LH_gecko"), "command_name": "LH_pose"},
    )
    RF_orient_tracking = RewTerm(
        func=mdp.orientation_command_error_ln,
        weight=weight_orient_track,
        params={"epsilon": epsilon_orient_track,"asset_cfg": SceneEntityCfg("robot", body_names=".*RF_gecko"), "command_name": "RF_pose"},
    )
    RH_orient_tracking = RewTerm(
        func=mdp.orientation_command_error_ln,
        weight=weight_orient_track,
        params={"epsilon": epsilon_orient_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*RH_gecko"), "command_name": "RH_pose"},
    )
    
    
    
    # -- penalties
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.0)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-6)
    dof_power_l2 = RewTerm(func=mdp.joint_power_l2, weight=-0.005)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0)
    body_lin_acc = RewTerm(func=mdp.body_lin_acc_l2,
                           weight=-1.0e-3,
                           params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*body", ".*gecko"])})
    body_ang_acc = RewTerm(func=mdp.body_ang_acc_l2,
                           weight=-1.0e-3,
                           params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*body", ".*gecko"])})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*shoulder_link", ".*upper_arm_link", ".*forearm_link", ".*wrist_1_link", ".*wrist_2_link", ".*wrist_3_link"]), "threshold": 1.0}, 
    )
    # -- termination penalties
    illegal_contact = RewTerm(
        func=mdp.illegal_contact,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*body"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*body"), "threshold": 1.0},
    )
    '''
    no_feet_contact = DoneTerm(
        func=mdp.feet_contact_num,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*gecko"]), "threshold": 1},
    )
    '''



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate_l2", "weight": -0.005, "num_steps": 4500}
    )


##
# Environment configuration
##


@configclass
class LegPositionControlEnvCfg(RLTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 8
        self.episode_length_s = 12.0
        # simulation settings
        self.sim.dt = 0.0025
        self.sim.gravity = (0.0, 0.0, 0.0) #remove gravity
        self.sim.disable_contact_processing = True