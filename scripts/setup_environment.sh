#!/bin/bash

# Copyright (c) 2025 IsaacSimG1 Project.
# SPDX-License-Identifier: Apache-2.0

#==
# IsaacSimG1 Environment Setup Script
# This script sets up the complete environment for Isaac Sim, Isaac Lab, and G1 locomotion training
#==

# Exit on any error
set -e

# Set tab-spaces
tabs 4

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the root directory of the project
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
export ISAACLAB_PATH="${PROJECT_ROOT}/external/IsaacLab"
export ISAACSIM_PATH="${PROJECT_ROOT}/external/IsaacSim"
export G1_PROJECT_PATH="${PROJECT_ROOT}/external/g1_23dof_locomotion_isaac"

# Default environment name
DEFAULT_ENV_NAME="isaaclab"

#==
# Helper Functions
#==

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}===================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}===================================================${NC}\n"
}

# Check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda could not be found. Please install conda and try again."
        echo "You can download conda from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_success "Conda is installed"
}

# Check if docker is installed (for Isaac Sim)
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Some Isaac Sim features may not work."
        print_info "You can install docker from: https://docs.docker.com/get-docker/"
    else
        print_success "Docker is installed"
    fi
}

# Setup conda environment
setup_conda_env() {
    local env_name=${1:-$DEFAULT_ENV_NAME}
    
    print_header "Setting up Conda Environment: $env_name"
    
    # Check if the environment exists
    if { conda env list | grep -w ${env_name}; } >/dev/null 2>&1; then
        print_info "Conda environment '${env_name}' already exists."
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing environment..."
            conda env remove -n ${env_name} -y
        else
            print_info "Using existing environment."
            return 0
        fi
    fi
    
    print_info "Creating conda environment '${env_name}'..."
    
    # Create environment from Isaac Lab's environment.yml
    if [ -f "${ISAACLAB_PATH}/environment.yml" ]; then
        conda env create -y --file ${ISAACLAB_PATH}/environment.yml -n ${env_name}
    else
        print_warning "Isaac Lab environment.yml not found. Creating basic environment..."
        conda create -y -n ${env_name} python=3.10
    fi
    
    print_success "Conda environment '${env_name}' created successfully"
}

# Install Isaac Sim dependencies
install_isaacsim_deps() {
    print_header "Installing Isaac Sim Dependencies"
    
    # Activate the conda environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${env_name}
    
    # Install PyTorch (required for Isaac Sim)
    print_info "Installing PyTorch..."
    conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    
    # Install Isaac Sim pip package if available
    print_info "Installing Isaac Sim pip package..."
    pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit || {
        print_warning "Isaac Sim pip packages not available. Will use local installation."
    }
    
    print_success "Isaac Sim dependencies installed"
}

# Install Isaac Lab
install_isaaclab() {
    print_header "Installing Isaac Lab"
    
    # Activate the conda environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${env_name}
    
    # Install Isaac Lab and related packages
    print_info "Installing Isaac Lab packages..."
    
    # Install core Isaac Lab package
    if [ -f "${ISAACLAB_PATH}/source/isaaclab/setup.py" ]; then
        pip install -e "${ISAACLAB_PATH}/source/isaaclab"
    fi
    
    # Install Isaac Lab assets
    if [ -f "${ISAACLAB_PATH}/source/isaaclab_assets/setup.py" ]; then
        pip install -e "${ISAACLAB_PATH}/source/isaaclab_assets"
    fi
    
    # Install Isaac Lab tasks
    if [ -f "${ISAACLAB_PATH}/source/isaaclab_tasks/setup.py" ]; then
        pip install -e "${ISAACLAB_PATH}/source/isaaclab_tasks"
    fi
    
    # Install Isaac Lab RL
    if [ -f "${ISAACLAB_PATH}/source/isaaclab_rl/setup.py" ]; then
        pip install -e "${ISAACLAB_PATH}/source/isaaclab_rl"
    fi
    
    # Install Isaac Lab mimic
    if [ -f "${ISAACLAB_PATH}/source/isaaclab_mimic/setup.py" ]; then
        pip install -e "${ISAACLAB_PATH}/source/isaaclab_mimic"
    fi
    
    print_success "Isaac Lab installed successfully"
}

# Install G1 locomotion project
install_g1_project() {
    print_header "Installing G1 Locomotion Project"
    
    # Activate the conda environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${env_name}
    
    # Install G1 locomotion package
    if [ -f "${G1_PROJECT_PATH}/source/g1_23dof_locomotion_isaac/setup.py" ]; then
        print_info "Installing G1 locomotion package..."
        pip install -e "${G1_PROJECT_PATH}/source/g1_23dof_locomotion_isaac"
    else
        print_warning "G1 locomotion setup.py not found"
    fi
    
    # Install additional dependencies
    print_info "Installing additional dependencies..."
    pip install rsl-rl-lib==2.3.1 rl-games==1.6.1 ray[rllib]==2.6.3 wandb tensorboard matplotlib
    
    # Install MuJoCo for sim2sim deployment
    print_info "Installing MuJoCo..."
    pip install mujoco dm_control
    
    print_success "G1 locomotion project installed successfully"
}

# Setup WebRTC for livestreaming (optional)
setup_webrtc() {
    print_header "Setting up WebRTC for Livestreaming (Optional)"
    
    print_info "WebRTC allows real-time video monitoring of Isaac Sim training sessions."
    print_info "This is useful for visual debugging and monitoring robot behavior."
    
    # Check if user wants to install WebRTC dependencies
    print_warning "WebRTC setup requires system packages and may need sudo privileges."
    read -p "Do you want to install WebRTC dependencies? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installing WebRTC system dependencies..."
        
        # Install required system libraries
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y libxt6 libxrandr2 libglu1-mesa libgl1-mesa-glx
            print_success "WebRTC system dependencies installed"
        elif command -v yum &> /dev/null; then
            sudo yum install -y libXt libXrandr2 mesa-libGLU
            print_success "WebRTC system dependencies installed"
        else
            print_warning "Package manager not recognized. Please install manually:"
            print_info "  Required packages: libXt6, libXrandr2, libGLU1-mesa"
        fi
        
        # Create WebRTC configuration note
        print_info "Creating WebRTC configuration notes..."
        cat > ${PROJECT_ROOT}/WEBRTC_SETUP.md << EOF
# WebRTC Livestreaming Setup

## Overview
WebRTC allows real-time video streaming of Isaac Sim training sessions for monitoring and debugging.

## Requirements
- System packages: libXt6, libXrandr2, libGLU1-mesa
- Isaac Sim WebRTC extensions (automatically loaded)
- Network ports: 8211 (HTTP), 49100 (UDP)

## Usage

### Training with WebRTC
\`\`\`bash
# Start training with WebRTC livestreaming
g1_train --task=Loco --livestream=2

# Or using environment variables
export LIVESTREAM=2
g1_train --task=Loco
\`\`\`

### Playing with WebRTC
\`\`\`bash
# Play trained policy with WebRTC visualization
g1_play --task=Loco --checkpoint=/path/to/model.pt --livestream=2 --num_envs=16
\`\`\`

### Accessing WebRTC Stream
- Open browser to: http://localhost:8211/streaming/webrtc-demo/
- Or use the Isaac Sim Streaming Client application

## Environment Variables
- LIVESTREAM=2: Enable WebRTC streaming
- OMNI_WEBRTC_PORT_HTTP=8211: HTTP port for WebRTC
- OMNI_WEBRTC_PORT_UDP=49100: UDP port for WebRTC

## Troubleshooting
- If segmentation fault occurs, ensure all system libraries are installed
- Check firewall settings for ports 8211 and 49100
- For headless servers, WebRTC may conflict with display requirements

## Notes
- WebRTC streaming may impact training performance
- Use fewer environments (--num_envs=4-16) for better streaming quality
- Headless training is recommended for production, WebRTC for debugging
EOF
        
        print_success "WebRTC setup completed!"
        print_info "WebRTC configuration notes saved to: ${PROJECT_ROOT}/WEBRTC_SETUP.md"
    else
        print_info "Skipping WebRTC setup. You can run this later by calling:"
        print_info "  setup_webrtc"
    fi
}

# Setup environment variables and aliases
setup_environment() {
    print_header "Setting up Environment Variables and Aliases"
    
    local env_name=${1:-$DEFAULT_ENV_NAME}
    
    # Get conda prefix
    local conda_prefix=$(conda info --base)/envs/${env_name}
    
    # Create conda activation script
    mkdir -p ${conda_prefix}/etc/conda/activate.d
    mkdir -p ${conda_prefix}/etc/conda/deactivate.d
    
    # Create activation script
    cat > ${conda_prefix}/etc/conda/activate.d/isaaclab_env.sh << EOF
#!/usr/bin/env bash

# Isaac Lab Environment Setup
export ISAACLAB_PATH="${ISAACLAB_PATH}"
export ISAACSIM_PATH="${ISAACSIM_PATH}"
export G1_PROJECT_PATH="${G1_PROJECT_PATH}"
export PROJECT_ROOT="${PROJECT_ROOT}"

# Isaac Lab alias
alias isaaclab="${ISAACLAB_PATH}/isaaclab.sh"

# G1 project aliases
alias g1_train="cd ${G1_PROJECT_PATH} && python scripts/rsl_rl/train.py"
alias g1_play="cd ${G1_PROJECT_PATH} && python scripts/rsl_rl/play.py"
alias g1_deploy="cd ${G1_PROJECT_PATH}/deployment && python deploy_sim.py"

# WebRTC streaming aliases
alias g1_train_webrtc="cd ${G1_PROJECT_PATH} && python scripts/rsl_rl/train.py --livestream=2"
alias g1_play_webrtc="cd ${G1_PROJECT_PATH} && python scripts/rsl_rl/play.py --livestream=2"

# Set resource name for Isaac Sim
export RESOURCE_NAME="IsaacSim"

# Add project paths to Python path
export PYTHONPATH="${G1_PROJECT_PATH}:${ISAACLAB_PATH}/source:${PYTHONPATH}"

echo "Isaac Lab G1 environment activated!"
echo "Available aliases:"
echo "  isaaclab        - Run Isaac Lab"
echo "  g1_train        - Train G1 locomotion"
echo "  g1_play         - Play trained G1 policy"
echo "  g1_deploy       - Deploy to MuJoCo"
echo "  g1_train_webrtc - Train with WebRTC livestreaming"
echo "  g1_play_webrtc  - Play with WebRTC livestreaming"
EOF

    # Create deactivation script
    cat > ${conda_prefix}/etc/conda/deactivate.d/isaaclab_env.sh << EOF
#!/usr/bin/env bash

# Unset environment variables
unset ISAACLAB_PATH
unset ISAACSIM_PATH
unset G1_PROJECT_PATH
unset PROJECT_ROOT
unset RESOURCE_NAME

# Remove aliases
unalias isaaclab 2>/dev/null || true
unalias g1_train 2>/dev/null || true
unalias g1_play 2>/dev/null || true
unalias g1_deploy 2>/dev/null || true
unalias g1_train_webrtc 2>/dev/null || true
unalias g1_play_webrtc 2>/dev/null || true

echo "Isaac Lab G1 environment deactivated!"
EOF

    # Make scripts executable
    chmod +x ${conda_prefix}/etc/conda/activate.d/isaaclab_env.sh
    chmod +x ${conda_prefix}/etc/conda/deactivate.d/isaaclab_env.sh
    
    print_success "Environment setup completed"
}

# Create Isaac Sim symlink
create_isaacsim_symlink() {
    print_header "Creating Isaac Sim Symlink"
    
    local isaac_sim_symlink="${ISAACLAB_PATH}/_isaac_sim"
    
    if [ -L "$isaac_sim_symlink" ]; then
        print_info "Isaac Sim symlink already exists"
        return 0
    fi
    
    # Try to find Isaac Sim installation
    local isaac_sim_paths=(
        "/opt/nvidia/isaacsim"
        "$HOME/.local/share/ov/pkg/isaac_sim"
        "$HOME/Downloads/isaac_sim"
        "${ISAACSIM_PATH}"
    )
    
    local isaac_sim_found=""
    for path in "${isaac_sim_paths[@]}"; do
        if [ -d "$path" ]; then
            isaac_sim_found="$path"
            break
        fi
    done
    
    if [ -n "$isaac_sim_found" ]; then
        print_info "Creating symlink to Isaac Sim at: $isaac_sim_found"
        ln -s "$isaac_sim_found" "$isaac_sim_symlink"
        print_success "Isaac Sim symlink created"
    else
        print_warning "Isaac Sim installation not found automatically."
        print_info "Please create a symlink manually:"
        print_info "  ln -s /path/to/isaac_sim ${isaac_sim_symlink}"
    fi
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    local env_name=${1:-$DEFAULT_ENV_NAME}
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${env_name}
    
    # Check Python packages
    print_info "Checking Python packages..."
    python -c "import isaaclab; print('Isaac Lab: OK')" || print_error "Isaac Lab not found"
    python -c "import torch; print('PyTorch: OK')" || print_error "PyTorch not found"
    python -c "import gymnasium; print('Gymnasium: OK')" || print_error "Gymnasium not found"
    
    # Check G1 project
    python -c "import g1_23dof_locomotion_isaac; print('G1 Project: OK')" || print_warning "G1 project not found"
    
    print_success "Installation verification completed"
}

# Main function
main() {
    print_header "Isaac Sim G1 Environment Setup"
    
    # Parse arguments
    local env_name=${1:-$DEFAULT_ENV_NAME}
    local skip_verification=${2:-false}
    
    print_info "Project root: $PROJECT_ROOT"
    print_info "Environment name: $env_name"
    print_info "Isaac Lab path: $ISAACLAB_PATH"
    print_info "Isaac Sim path: $ISAACSIM_PATH"
    print_info "G1 project path: $G1_PROJECT_PATH"
    
    # Check prerequisites
    check_conda
    check_docker
    
    # Setup conda environment
    setup_conda_env "$env_name"
    
    # Install packages
    install_isaacsim_deps
    install_isaaclab
    install_g1_project
    
    # Setup environment
    setup_environment "$env_name"
    
    # Create Isaac Sim symlink
    create_isaacsim_symlink
    
    # Verify installation
    if [ "$skip_verification" != "true" ]; then
        verify_installation "$env_name"
    fi
    
    # Setup WebRTC
    setup_webrtc
    
    print_header "Setup Complete!"
    print_success "Isaac Lab G1 environment setup completed successfully!"
    print_info "To activate the environment, run:"
    print_info "  conda activate $env_name"
    print_info ""
    print_info "To start training:"
    print_info "  g1_train --task=Loco --headless"
    print_info ""
    print_info "To play a trained policy:"
    print_info "  g1_play --task Loco --num_envs 32"
    print_info ""
    print_info "To deploy to MuJoCo:"
    print_info "  g1_deploy"
    print_info ""
    print_info "For WebRTC livestreaming (if enabled):"
    print_info "  g1_train_webrtc --task=Loco --num_envs=16"
    print_info "  g1_play_webrtc --task=Loco --checkpoint=/path/to/model.pt"
    print_info "  Then open: http://localhost:8211/streaming/webrtc-demo/"
}

# Show help
show_help() {
    echo "Usage: $0 [ENV_NAME] [--skip-verification]"
    echo ""
    echo "Arguments:"
    echo "  ENV_NAME           Name of the conda environment (default: isaaclab)"
    echo "  --skip-verification Skip the installation verification step"
    echo ""
    echo "Examples:"
    echo "  $0                 # Setup with default environment name"
    echo "  $0 my_env          # Setup with custom environment name"
    echo "  $0 isaaclab --skip-verification  # Skip verification"
}

# Parse command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Run main function
main "$@"
