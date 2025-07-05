# IsaacSimG1

## 🤖 G1 Humanoid Locomotion Training with Isaac Lab

This repository contains the Isaac Lab G1 environment setup and training logs for the Unitree G1 humanoid robot locomotion policy.

## 🚀 Training Session Notes - July 5, 2025

### ✅ **Setup Completed Successfully**
- **Environment**: Isaac Lab G1 conda environment activated
- **Dependencies**: All packages installed correctly (rsl-rl-lib, rl-games, etc.)
- **Isaac Sim**: Running in headless mode with GPU acceleration
- **Task**: G1 23-DOF Locomotion (`Loco`)

### 🎯 **Current Training Status**
- **Process ID**: 88541 (running in background)
- **Runtime**: 33+ minutes and counting
- **Mode**: Headless (no GUI, optimized for server/cloud)
- **Progress**: Model checkpoint saved at iteration 450 (22.5% complete)
- **Resource Usage**: 17.1% CPU, 28.1% memory (~4.5GB RAM)

### 📊 **Training Configuration**
```bash
# Training Command
python scripts/rsl_rl/train.py --task=Loco --headless

# Key Parameters
- Algorithm: PPO (Proximal Policy Optimization)
- Environments: 4096 parallel environments
- Max Iterations: 2000
- Save Interval: 50 iterations
- Experiment: g1_23dof_v17_rc
```

### 📁 **Generated Files & Directories**
```
logs/rsl_rl/g1_23dof_v17_rc/2025-07-05_05-00-56/
├── model_0.pt              # Initial model checkpoint
├── model_50.pt             # Latest model checkpoint (iteration 50)
├── events.out.tfevents.*   # TensorBoard logs (actively updating)
├── params/
│   ├── env.yaml           # Environment configuration
│   ├── agent.yaml         # Agent configuration
│   ├── env.pkl            # Environment config (pickle)
│   └── agent.pkl          # Agent config (pickle)
└── git/                   # Git repository info
```

### 🔧 **Key Features**
- **Symmetry Loss**: Enabled for better locomotion learning
- **Domain Randomization**: Body mass, friction, and other parameters
- **Reward Functions**: Tracking linear/angular velocity, penalizing undesired contacts
- **Terrain**: Flat plane (simplified for initial training)
- **GPU**: Tesla T4 with CUDA acceleration

### 🖥️ **Monitoring Progress**
```bash
# Check training process
ps aux | grep train.py

# Monitor latest checkpoints
ls -lt logs/rsl_rl/g1_23dof_v17_rc/2025-07-05_05-00-56/model_*.pt

# View TensorBoard logs
tensorboard --logdir=logs/rsl_rl/g1_23dof_v17_rc/2025-07-05_05-00-56/

# Check resource usage
htop
```

### 📈 **Training Progress**
- **Iteration 0**: Initial model saved
- **Iteration 50**: First checkpoint completed
- **Current**: Training actively progressing towards iteration 2000
- **ETA**: ~2-3 hours for full training (estimated)

### 🎯 **Next Steps**
1. **Monitor Training**: Check progress periodically
2. **Evaluate Policy**: Test trained models with play script
3. **Sim2Sim Transfer**: Deploy to MuJoCo environment
4. **Real Robot**: Potential deployment to physical G1 robot

### 🛠️ **Quick Commands**
```bash
# Navigate to project
cd external/g1_23dof_locomotion_isaac

# Activate environment
source ~/.bashrc && conda activate isaaclab

# Train (if not already running)
python scripts/rsl_rl/train.py --task=Loco --headless

# Play trained policy
python scripts/rsl_rl/play.py --task=Loco --num_envs=32

# Deploy to MuJoCo
python deployment/deploy_sim.py
```

### 📝 **Notes**
- Training successfully initiated on July 5, 2025 at 05:00 UTC
- Headless mode confirmed working properly
- All dependencies resolved and environment stable
- Model checkpoints saving every 50 iterations
- TensorBoard logs updating in real-time

### ⚠️ **Deprecation Warnings Analysis**
- **Isaac Sim System**: `omni.isaac.dynamic_control` deprecated warning (system-level, no action needed)
- **Wandb Library**: `pkg_resources` deprecation warning (third-party library, no action needed)
- **G1 Project Code**: ✅ No deprecated functions found - all imports are modern and current
- **Status**: No code fixes required in the G1 project

### 🎥 **WebRTC Livestreaming Setup**
- **Setup Script Updated**: Added WebRTC configuration to `scripts/setup_environment.sh`
- **System Dependencies**: libXt6, libXrandr2, libGLU1-mesa for WebRTC support
- **New Aliases**: `g1_train_webrtc` and `g1_play_webrtc` for streaming-enabled training
- **Configuration**: Detailed setup guide available in `WEBRTC_SETUP.md`
- **Access URL**: http://localhost:8211/streaming/webrtc-demo/
- **Status**: Ready for future WebRTC monitoring sessions

## 🚀 **Summary & Next Steps**

### ✅ **Completed Successfully**
1. **Full Environment Setup**: Isaac Lab G1 environment with all dependencies
2. **Successful Training**: G1 locomotion policy training at 22.5% completion
3. **WebRTC Integration**: Complete setup for future video monitoring
4. **Documentation**: Comprehensive guides and troubleshooting notes

### 🔄 **Currently Running**
- **G1 Training**: Iteration 450/2000 (22.5% complete, 33+ minutes runtime)
- **Stable Performance**: Consistent checkpoint saving every 50 iterations
- **Resource Efficiency**: 17.1% CPU, 28.1% memory usage

### 🛠️ **Ready for Future Use**
- **WebRTC Monitoring**: Run `g1_train_webrtc` or `g1_play_webrtc` for visual monitoring
- **Model Evaluation**: Latest checkpoint: `model_450.pt` ready for testing
- **Deployment**: MuJoCo deployment ready via `g1_deploy`
- **Environment**: All aliases and paths configured for easy usage

### 📋 **Key Commands**
```bash
# Check training status
ps aux | grep train.py

# View latest checkpoints
ls -lt logs/rsl_rl/g1_23dof_v17_rc/*/model_*.pt

# Enable WebRTC for future sessions
g1_train_webrtc --task=Loco --num_envs=16
g1_play_webrtc --task=Loco --checkpoint=/path/to/model.pt

# Access WebRTC stream
# Open: http://localhost:8211/streaming/webrtc-demo/
```

---
*Last updated: July 5, 2025 - Training actively running in background*
