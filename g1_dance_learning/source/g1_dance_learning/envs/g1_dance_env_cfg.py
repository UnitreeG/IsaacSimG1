# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 dance imitation using motion capture data."""

import math
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Import MDP functions
import isaaclab.envs.mdp as mdp

# Import G1 robot configuration
try:
    from isaaclab_assets import G1_MINIMAL_CFG
    G1_CFG = G1_MINIMAL_CFG
except ImportError:
    # Fallback if isaaclab_assets is not available
    from isaaclab.assets import ArticulationCfg
    G1_CFG = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/Isaac/Robots/Unitree/G1/g1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={
                ".*hip_yaw_joint": 0.0,
                ".*hip_roll_joint": 0.0,
                ".*hip_pitch_joint": -0.15,
                ".*knee_joint": 0.3,
                ".*ankle_pitch_joint": -0.15,
                ".*ankle_roll_joint": 0.0,
                ".*shoulder_pitch_joint": 0.0,
                ".*shoulder_roll_joint": 0.0,
                ".*shoulder_yaw_joint": 0.0,
                ".*elbow_joint": 0.0,
                ".*wrist_roll_joint": 0.0,
                "waist_yaw_joint": 0.0,
            },
        ),
        actuators={
            "body": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=300.0,
                velocity_limit=100.0,
                stiffness=25.0,
                damping=0.5,
            ),
        },
    )

# Motion files directory
MOTIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "motions")

##
# Scene definition
##

@configclass
class G1DanceSceneCfg(InteractiveSceneCfg):
    """Configuration for the G1 dance imitation scene."""

    # ground terrain - flat for dance performance
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = MISSING

    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True
    )

    # lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

##
# MDP definitions
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.3,  # Reduced scale for more precise dance movements
        use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot state observations
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # Joint state observations
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        
        # Action history
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # Same as policy but without noise
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: ObsGroup = PolicyCfg()
    critic: ObsGroup = CriticCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # reset robot to default pose
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    # reset base to origin with small variation
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Basic stability rewards
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=1.0,
        params={"target_height": 0.98}
    )
    
    # Balance reward
    feet_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"), 
            "threshold": 0.1
        },
    )

    # Smoothness rewards
    action_smoothness = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01
    )
    
    joint_acc_penalty = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7
    )

    # Penalties
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-50.0
    )

    # Avoid unnatural joint limits
    joint_limits_penalty = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Terminate if robot falls
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass  # No curriculum for basic setup

##
# Environment configuration
##

@configclass
class G1DanceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for G1 dance imitation environment."""

    # Scene settings
    scene: G1DanceSceneCfg = G1DanceSceneCfg(num_envs=1024, env_spacing=4.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4  # 50 Hz control
        self.episode_length_s = 20.0  # Episode length
        
        # simulation settings
        self.sim.dt = 0.005  # 200 Hz simulation
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        
        # robot settings
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # update sensor update periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        
        # Disable terrain complexity for dance
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Customize reset behavior for dance
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.05, 0.05)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        
        # Reduce randomization for more consistent learning
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        
        # Termination settings
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link" 