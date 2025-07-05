#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Train G1 robot to dance using motion imitation with RSL-RL.
"""

import argparse
import os
import sys

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

# Add the g1_dance_learning package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

import g1_dance_learning.tasks  # noqa: F401
from g1_dance_learning.envs.g1_dance_env_cfg import G1DanceEnvCfg

"""
Configuration for RSL-RL training.
"""

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnvWrapper

# Define the RSL-RL configuration
from rsl_rl.algorithms import PPOConfig
from rsl_rl.utils.logging import Logger

class G1DancePPOConfig(PPOConfig):
    """Configuration for G1 dance imitation PPO."""
    
    def __init__(self):
        super().__init__()
        
        # PPO algorithm configuration
        self.clip_param = 0.2
        self.num_learning_epochs = 5
        self.num_mini_batches = 4
        self.learning_rate = 3e-4
        self.schedule = "adaptive"
        self.gamma = 0.99
        self.lam = 0.95
        self.desired_kl = 0.01
        self.max_grad_norm = 1.0
        self.entropy_coef = 0.0
        self.learning_rate_schedule = "linear"
        
        # Network configuration
        self.policy_hidden_dims = [512, 256, 128]
        self.value_hidden_dims = [512, 256, 128]
        self.activation = "elu"
        
        # Training configuration
        self.num_steps_per_env = 24
        self.max_iterations = 20000
        self.save_interval = 50
        
        # Logging
        self.log_interval = 10
        self.experiment_name = "g1_dance_learning"
        
        # Device
        self.device = "cuda"
        
        # Normalization
        self.use_clipped_value_loss = True
        self.normalize_advantage = True

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train G1 robot to dance using motion imitation with RSL-RL."
    )
    parser.add_argument(
        "--env_cfg",
        type=str,
        default="G1-Dance-v0",
        help="Environment configuration name.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
        help="Number of environments to use.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=20000,
        help="Maximum number of training iterations.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training.",
    )
    args = parser.parse_args()
    
    # Create environment configuration
    env_cfg = G1DanceEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    
    # Create environment
    import gymnasium as gym
    env = gym.make(args.env_cfg, cfg=env_cfg)
    
    # Wrap environment for RSL-RL
    env = VecEnvWrapper(env)
    
    # Create RSL-RL configuration
    rsl_rl_cfg = G1DancePPOConfig()
    rsl_rl_cfg.device = args.device
    rsl_rl_cfg.max_iterations = args.max_iterations
    
    # Set random seed
    if args.seed is not None:
        env.seed(args.seed)
    
    # Create runner
    runner = OnPolicyRunner(env, rsl_rl_cfg, log_dir=None, device=args.device)
    
    # Resume from checkpoint if provided
    if args.resume_checkpoint is not None:
        runner.load(args.resume_checkpoint)
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"  Environment: {args.env_cfg}")
    print(f"  Number of environments: {args.num_envs}")
    print(f"  Maximum iterations: {args.max_iterations}")
    print(f"  Device: {args.device}")
    print(f"  Seed: {args.seed}")
    
    # Start training
    print("\nStarting training...")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    # Save final policy
    print("\nSaving final policy...")
    runner.save(os.path.join(runner.log_dir, "final_policy"))
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main() 