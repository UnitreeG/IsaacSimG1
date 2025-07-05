#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training script for G1 dance imitation using RSL-RL."""

import argparse
import os
import sys

# Add the G1 extension to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "external", "g1_23dof_locomotion_isaac", "source"))

def main():
    """Main training function."""
    
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Train G1 robot to perform dance imitation")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
    parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training")
    parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings")
    parser.add_argument("--max_iterations", type=int, default=5000, help="Maximum training iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    
    args = parser.parse_args()
    
    # Import Isaac Lab modules (after argument parsing to handle potential import issues)
    try:
        from isaaclab.app import AppLauncher
        
        # Create app launcher configuration
        app_launcher_cfg = AppLauncher.AppLauncherCfg()
        app_launcher_cfg.headless = args.headless
        app_launcher_cfg.enable_cameras = args.video
        
        # Launch Isaac Sim
        app_launcher = AppLauncher(app_launcher_cfg)
        simulation_app = app_launcher.app
        
        # Import Isaac Lab components
        import gymnasium as gym
        import torch
        from datetime import datetime
        
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.envs import ManagerBasedRLEnvCfg
        from isaaclab.utils.dict import print_dict
        from isaaclab.utils.io import dump_pickle, dump_yaml
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import get_checkpoint_path
        
        # Import G1 tasks
        import g1_23dof_locomotion_isaac.tasks
        
        print("=== G1 Dance Imitation Training ===")
        print(f"Number of environments: {args.num_envs}")
        print(f"Training device: {args.device}")
        print(f"Max iterations: {args.max_iterations}")
        print(f"Random seed: {args.seed}")
        
        # Set random seeds
        torch.manual_seed(args.seed)
        
        # Create environment
        env_cfg = gym.make("G1-Dance", num_envs=args.num_envs, render_mode="rgb_array" if args.video else None)
        
        # Configure environment settings
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device
        env_cfg.seed = args.seed
        
        # Create environment
        env = gym.make("G1-Dance", cfg=env_cfg)
        
        # Configure RSL-RL
        rsl_rl_cfg = RslRlOnPolicyRunnerCfg()
        rsl_rl_cfg.seed = args.seed
        rsl_rl_cfg.device = args.device
        rsl_rl_cfg.max_iterations = args.max_iterations
        rsl_rl_cfg.experiment_name = "g1_dance_imitation"
        
        # Customize training parameters for dance imitation
        rsl_rl_cfg.algorithm.value_loss_coef = 1.0
        rsl_rl_cfg.algorithm.use_clipped_value_loss = True
        rsl_rl_cfg.algorithm.clip_param = 0.2
        rsl_rl_cfg.algorithm.entropy_coef = 0.01
        rsl_rl_cfg.algorithm.num_learning_epochs = 5
        rsl_rl_cfg.algorithm.num_mini_batches = 4
        rsl_rl_cfg.algorithm.learning_rate = 1.0e-3
        rsl_rl_cfg.algorithm.schedule = "adaptive"
        rsl_rl_cfg.algorithm.gamma = 0.99
        rsl_rl_cfg.algorithm.lam = 0.95
        rsl_rl_cfg.algorithm.desired_kl = 0.01
        rsl_rl_cfg.algorithm.max_grad_norm = 1.0
        
        # Set up logging
        log_root_path = os.path.join("logs", "rsl_rl", rsl_rl_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"Logging experiment in directory: {log_root_path}")
        
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root_path, log_dir)
        
        # Add video recording if requested
        if args.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args.video_interval == 0,
                "video_length": args.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        
        # Wrap environment for RSL-RL
        env = RslRlVecEnvWrapper(env)
        
        # Create runner
        runner = OnPolicyRunner(env, rsl_rl_cfg.to_dict(), log_dir=log_dir, device=args.device)
        
        # Save configuration
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), rsl_rl_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), rsl_rl_cfg)
        
        print("\n=== Starting Training ===")
        print("Press Ctrl+C to stop training and save the model")
        
        # Run training
        runner.learn(num_learning_iterations=rsl_rl_cfg.max_iterations, init_at_random_ep_len=True)
        
        print("\n=== Training Complete ===")
        print(f"Model saved to: {log_dir}")
        
        # Close environment
        env.close()
        
    except KeyboardInterrupt:
        print("\n=== Training Interrupted ===")
        print("Saving current model...")
        if 'runner' in locals():
            runner.save(os.path.join(log_dir, "model_interrupted.pt"))
        print(f"Model saved to: {log_dir}")
        
    except ImportError as e:
        print(f"Error importing Isaac Lab modules: {e}")
        print("Please make sure you're running this script from within the Isaac Lab environment.")
        print("You can activate the environment with: source ~/.bashrc && conda activate isaaclab")
        sys.exit(1)
        
    finally:
        if 'simulation_app' in locals():
            simulation_app.close()


if __name__ == "__main__":
    main() 