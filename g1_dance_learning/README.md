# G1 Dance Learning

This project implements a reinforcement learning system for teaching the G1 humanoid robot to dance using motion imitation techniques with Isaac Lab.

## Project Structure

```
g1_dance_learning/
├── source/
│   └── g1_dance_learning/
│       ├── __init__.py
│       ├── envs/
│       │   ├── __init__.py
│       │   └── g1_dance_env_cfg.py    # Environment configuration
│       ├── tasks/
│       │   └── __init__.py             # Task registration
│       └── utils/
│           ├── __init__.py
│           └── motion_loader.py        # Motion data loading utilities
├── scripts/
│   ├── bvh_to_npz_converter.py        # BVH to NPZ converter
│   └── train_g1_dance.py              # Training script
├── configs/                           # Configuration files
├── motions/                           # Motion data files
│   └── macarena_dance.npz            # Converted Macarena dance motion
├── logs/                             # Training logs
└── README.md                         # This file
```

## Features

- **Motion Imitation**: Learn to imitate dance movements from BVH motion capture data
- **G1 Robot Support**: Specifically designed for the Unitree G1 humanoid robot
- **Isaac Lab Integration**: Built on Isaac Lab framework for physics simulation
- **RSL-RL Training**: Uses RSL-RL for policy learning with PPO algorithm
- **Motion Data Conversion**: Converts BVH files to Isaac Lab's NPZ format

## Installation

1. Make sure you have Isaac Lab installed and set up
2. Ensure you have the required dependencies:
   - Isaac Lab
   - RSL-RL
   - PyTorch
   - NumPy
   - BVH parsing library

## Usage

### Converting BVH Files to NPZ

```bash
cd g1_dance_learning
python scripts/bvh_to_npz_converter.py --bvh_file path/to/dance.bvh --output_file motions/dance.npz --fps 30
```

### Training the Dance Policy

```bash
cd g1_dance_learning
python scripts/train_g1_dance.py --num_envs 1024 --max_iterations 20000 --device cuda
```

### Command Line Arguments

#### Training Script Options:
- `--env_cfg`: Environment configuration name (default: "G1-Dance-v0")
- `--num_envs`: Number of parallel environments (default: 1024)
- `--max_iterations`: Maximum training iterations (default: 20000)
- `--resume_checkpoint`: Path to checkpoint to resume from
- `--seed`: Random seed for reproducibility
- `--device`: Device to use (default: "cuda")

## Environment Configuration

The G1 dance environment includes:

- **Observation Space**: 
  - Base linear/angular velocity
  - Joint positions and velocities
  - Projected gravity vector
  - Action history

- **Action Space**: 
  - Joint position targets for all 23 DOF
  - Scaled actions for precise movement control

- **Reward Structure**:
  - Base height maintenance
  - Foot contact stability
  - Action smoothness
  - Joint limit penalties

## Motion Data Format

The system uses NPZ files with the following structure:
- `dof_names`: List of joint names
- `body_names`: List of body names
- `dof_positions`: Joint positions over time
- `dof_velocities`: Joint velocities over time
- `body_positions`: Body positions over time
- `body_rotations`: Body rotations (quaternions) over time
- `body_linear_velocities`: Body linear velocities over time
- `body_angular_velocities`: Body angular velocities over time
- `fps`: Frame rate of the motion data

## Current Motion Data

The project includes:
- **Macarena Dance**: 37.90 seconds of Macarena dance motion (1137 frames at 30 FPS)
  - Located at: `motions/macarena_dance.npz`
  - Contains full-body motion mapping from BVH to G1's 23 DOF

## Training Details

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network Architecture**: 
  - Policy: [512, 256, 128] hidden layers
  - Value: [512, 256, 128] hidden layers
- **Training Parameters**:
  - Learning rate: 3e-4
  - Batch size: 24 steps per environment
  - Discount factor: 0.99
  - GAE lambda: 0.95

## Next Steps

1. **Motion Imitation Rewards**: Add specific motion imitation rewards that compare robot poses to reference motion
2. **Phase Tracking**: Implement motion phase tracking for temporal alignment
3. **Multiple Dances**: Add support for multiple dance sequences
4. **Real Robot Deployment**: Test policies on physical G1 robot
5. **Interactive Control**: Add ability to trigger different dance sequences

## Notes

- The environment is configured for flat terrain to focus on dance performance
- Action scale is reduced (0.3) for more precise dance movements
- Episodes are 20 seconds long to allow for complete dance sequences
- The system uses caching for efficient motion data loading 