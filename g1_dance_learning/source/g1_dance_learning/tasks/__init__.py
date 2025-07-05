# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task configurations for G1 dance learning."""

import gymnasium as gym

from isaaclab.utils import configclass

from g1_dance_learning.envs.g1_dance_env_cfg import G1DanceEnvCfg

##
# Register Gym environments.
##

@configclass
class G1DanceEnvCfg_PLAY(G1DanceEnvCfg):
    """Configuration for G1 dance imitation environment for playing."""
    
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        
        # reduce action scale for smoother actions
        self.actions.joint_pos.scale = 0.25
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        
        # disable randomization for play
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

##
# Register Gym environments.
##

gym.register(
    id="G1-Dance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": G1DanceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{G1DanceEnvCfg.__module__}.{G1DanceEnvCfg.__qualname__}",
    },
)

gym.register(
    id="G1-Dance-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": G1DanceEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{G1DanceEnvCfg_PLAY.__module__}.{G1DanceEnvCfg_PLAY.__qualname__}",
    },
) 