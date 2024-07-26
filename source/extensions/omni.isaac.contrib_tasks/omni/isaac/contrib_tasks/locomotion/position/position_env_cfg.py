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
import omni.isaac.contrib_tasks.locomotion.position.mdp as mdp

# EE actions
from omni.isaac.contrib_tasks.leg_position_control.mdp.actions import actions_cfg
# EE observations
from omni.isaac.contrib_tasks.leg_position_control.mdp.observations import foot_position

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
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
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
    # robots
    robot: ArticulationCfg = TAKO_CFG
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/tako/body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/tako/.*", history_length=3, track_air_time=True, debug_vis=True)
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

    base_pose = mdp.UniformPose3dCommandCfg(
        asset_name="robot",
        body_name=".*body",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPose3dCommandCfg.Ranges(
            pos_x=(0.2, 0.5),
            pos_y=(0.0, 0.0),
            pos_z=MISSING, # Depends on init position
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

    LF_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=".*LF_dock",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(math.pi * 0.5, math.pi * 0.5),
            pitch=(math.pi * 0.5, math.pi * 0.5),  # depends on end-effector axis
            yaw=(0.0, 0.0),
        ),
    )
    LH_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=".*LH_dock",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(math.pi * 0.5, math.pi * 0.5),
            pitch=(math.pi * 0.5, math.pi * 0.5),  # depends on end-effector axis
            yaw=(0.0, 0.0),
        ),
    )
    RF_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=".*RF_dock",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(math.pi * 0.5, math.pi * 0.5),
            pitch=(math.pi * 0.5, math.pi * 0.5),  # depends on end-effector axis
            yaw=(0.0, 0.0),
        ),
    )    
    RH_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=".*RH_dock",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(math.pi * 0.5, math.pi * 0.5),
            pitch=(math.pi * 0.5, math.pi * 0.5),  # depends on end-effector axis
            yaw=(0.0, 0.0),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
    #joint_pos = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5)
    ''' 
    docking = actions_cfg.GripForceActionCfg(
        asset_name="robot",
        ee_names=[".*dock"],
        max_force=25.0,
        threshold=5.0,
    )

    
    LF_dock = actions_cfg.GripForceActionCfg(
        asset_name="robot",
        ee_names=[".*LF_dock"],
        max_force=25.0,
        threshold=5.0,
        command_name = "LF_pose",
    )
    LH_dock = actions_cfg.GripForceActionCfg(
        asset_name="robot",
        ee_names=[".*LH_dock"],
        max_force=25.0,
        threshold=5.0,
        command_name = "LH_pose",
    )
    RF_dock = actions_cfg.GripForceActionCfg(
        asset_name="robot",
        ee_names=[".*RF_dock"],
        max_force=25.0,
        threshold=5.0,
        command_name = "RF_pose",
    )
    RH_dock = actions_cfg.GripForceActionCfg(
        asset_name="robot",
        ee_names=[".*RH_dock"],
        max_force=25.0,
        threshold=5.0,
        command_name = "RH_pose",
    )
    '''
    
    body_thruster = actions_cfg.BodyThrusterActionCfg(
        asset_name="robot",
        max_push_force=20.0,
        threshold=1.0,
    )  



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        root_pos = ObsTerm(func=mdp.root_pos_w)  # 0 - 2
        #root_rot = ObsTerm(func=mdp.root_quat_w)  # 3 - 6
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # 7 - 9
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)  # 10 - 12
        projected_gravity = ObsTerm(func=mdp.projected_gravity)  # 13 - 15
        target_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_pose"})  # 16 - 22
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 23 - 46
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 47 - 70
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})  # 71 - 94
        #docking_action = ObsTerm(func=mdp.last_processed_action, params={"action_name": "docking"})  # 95 - 98
        body_thruster = ObsTerm(func=mdp.last_processed_action, params={"action_name": "body_thruster"})  # 99 - 101
        LF_foot_pos_des = ObsTerm(func=mdp.generated_commands, params={"command_name": "LF_pose"})  # 102 - 108
        LH_foot_pos_des = ObsTerm(func=mdp.generated_commands, params={"command_name": "LH_pose"})  # 109 - 115
        RF_foot_pos_des = ObsTerm(func=mdp.generated_commands, params={"command_name": "RF_pose"})  # 116 - 122
        RH_foot_pos_des = ObsTerm(func=mdp.generated_commands, params={"command_name": "RH_pose"})  # 123 - 129
        LF_foot_pos = ObsTerm(func=foot_position, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*LF_dock")})  # 130 - 132
        LH_foot_pos = ObsTerm(func=foot_position, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*LH_dock")})  # 133 - 135
        RF_foot_pos = ObsTerm(func=foot_position, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*RF_dock")})  # 136 - 138
        RH_foot_pos = ObsTerm(func=foot_position, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*RH_dock")})  # 139 - 141
        '''
        LF_dock = ObsTerm(func=mdp.desired_docking_state, params={"command_name": "LF_pose", "asset_cfg": SceneEntityCfg("robot", body_names=".*LF_dock")})  # 142
        LH_dock = ObsTerm(func=mdp.desired_docking_state, params={"command_name": "LH_pose", "asset_cfg": SceneEntityCfg("robot", body_names=".*LH_dock")})  # 143
        RF_dock = ObsTerm(func=mdp.desired_docking_state, params={"command_name": "RF_pose", "asset_cfg": SceneEntityCfg("robot", body_names=".*RF_dock")})  # 144
        RH_dock = ObsTerm(func=mdp.desired_docking_state, params={"command_name": "RH_pose", "asset_cfg": SceneEntityCfg("robot", body_names=".*RH_dock")})  # 145
        '''
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        #feet_contact = ObsTerm(func=mdp.feet_contacts, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*dock")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for Event."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    '''
    add_base_mass = EventTerm(
        func=mdp.add_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="body"), "mass_range": (-5.0, 5.0)},
    )
    '''

    # reset
    reset_base = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )    
    

weight_pos_track = 15.0
epsilon_pos_track = 1e-5
weight_orient_track = 15.0
epsilon_orient_track = 1e-5

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- BODY POSE TRACKING
    body_pose_tracking = RewTerm(
        func=mdp.base_pose_tracking_reward, 
        weight=25.0, 
        params={"command_name": "base_pose", "asset_cfg": SceneEntityCfg("robot", body_names=[".*body"])}
        )
    move_in_direction = RewTerm(
        func=mdp.move_in_direction_reward,
        weight=1.0,
        params={"command_name": "base_pose"}
    )
    '''
    body_orient_tracking = RewTerm(
        func=mdp.orientation_command_error_ln,
        weight=5.0,
        params={"command_name": "base_pose", "epsilon": 1e-5, "asset_cfg": SceneEntityCfg("robot", body_names=[".*body"])}
        )
        '''
    #move_in_direction = RewTerm(func=mdp.move_in_direction_reward, weight=5.0, params={"command_name": "base_pose"})

    LF_pos_tracking = RewTerm(
        func=mdp.position_command_error_ln,
        weight=weight_pos_track,
        params={"epsilon": epsilon_pos_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*LF_dock"), "command_name": "LF_pose"},
    )
    LH_pos_tracking = RewTerm(
        func=mdp.position_command_error_ln,
        weight=weight_pos_track,
        params={"epsilon": epsilon_pos_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*LH_dock"), "command_name": "LH_pose"},
    ) 
    RF_pos_tracking = RewTerm(
        func=mdp.position_command_error_ln,
        weight=weight_pos_track,
        params={"epsilon": epsilon_pos_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*RF_dock"), "command_name": "RF_pose"},
    ) 
    RH_pos_tracking = RewTerm(
        func=mdp.position_command_error_ln,
        weight=weight_pos_track,
        params={"epsilon": epsilon_pos_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*RH_dock"), "command_name": "RH_pose"},
    )  

    # ORIENTATION TRACKING
    LF_orient_tracking = RewTerm(
        func=mdp.orientation_command_error_ln,
        weight=weight_orient_track,
        params={"epsilon": epsilon_orient_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*LF_dock"), "command_name": "LF_pose"},
    )
    LH_orient_tracking = RewTerm(
        func=mdp.orientation_command_error_ln,
        weight=weight_orient_track,
        params={"epsilon": epsilon_orient_track,"asset_cfg": SceneEntityCfg("robot", body_names=".*LH_dock"), "command_name": "LH_pose"},
    )
    RF_orient_tracking = RewTerm(
        func=mdp.orientation_command_error_ln,
        weight=weight_orient_track,
        params={"epsilon": epsilon_orient_track,"asset_cfg": SceneEntityCfg("robot", body_names=".*RF_dock"), "command_name": "RF_pose"},
    )
    RH_orient_tracking = RewTerm(
        func=mdp.orientation_command_error_ln,
        weight=weight_orient_track,
        params={"epsilon": epsilon_orient_track, "asset_cfg": SceneEntityCfg("robot", body_names=".*RH_dock"), "command_name": "RH_pose"},
    )


    # -- penalties
    #dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0.0)
    #dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0.0)
    dof_power = RewTerm(func=mdp.joint_power_l2, weight=-7.5e-3)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-6)
    #default_dof_pos = RewTerm(func=mdp.joint_deviation_l1, weight=-1.0)
    #dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 0.95})
    #dof_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.2)
    #body_lin_acc = RewTerm(func=mdp.body_lin_acc_l2, weight=-1.0e-2, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*body"])})
    body_ang_acc = RewTerm(func=mdp.body_ang_acc_l2, weight=-1.0e-2, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*body"])})
    #feet_lin_acc = RewTerm(func=mdp.body_lin_acc_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*dock"])})
    #action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.0)
    action_term_rate_l2 = RewTerm(func=mdp.action_term_rate_l2, weight=-0.01, params={"term_name": "joint_pos"})
    #thruster_usage = RewTerm(func=mdp.action_term_l2, weight=-0., params={"action_name": "body_thruster"})
    #feet_contacts = RewTerm(func=mdp.feet_contacts, weight=1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*dock"])})
    #feet_xy_vel_in_contact = RewTerm(func=mdp.feet_xy_vel_in_contact, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*dock"]), "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*dock"])})
    #stand_at_target = RewTerm(func=mdp.stand_at_target, weight=-0., params={"command_name": "base_pose"})
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*upper_arm_link", ".*forearm_link", ".*wrist.*"]), "threshold": 1.0}, 
    )
    #stumble = RewTerm(func=mdp.stumble, weight=-0.0, params={"factor": 2.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*dock")})
    # Termination penalties
    feet_contact_num = RewTerm(func=mdp.feet_contact_num, weight=-0.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*dock"]), "threshold": 1})
    unhealthy_base_position = RewTerm(
        func=mdp.unhealthy_base_position,
        weight=-200.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*body"]), "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*body")},
        )
    # -- optional penalties
    #flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.0)
    #dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    #feet_power = RewTerm(func=mdp.feet_power, weight=-0.0, params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*dock"]), "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*dock")})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    unhealthy_base_position = DoneTerm(
        func=mdp.unhealthy_base_position,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*body"]), "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*body")},
        )
    '''
    no_feet_contact = DoneTerm(
        func=mdp.feet_contact_num,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*dock"]), "threshold": 1},
    )
    '''
    



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionPositionRoughEnvCfg(RLTaskEnvCfg):
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
        self.decimation = 4
        self.episode_length_s = 15.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.gravity = (0.0, 0.0, 0.0)
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # Adjust command resampling based on episode length
        self.commands.base_pose.resampling_time_range = (self.episode_length_s, self.episode_length_s)
        # Adjust rewards
        # For now, remove randomization events
        self.events.push_robot = None
        self.events.physics_material = None
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
