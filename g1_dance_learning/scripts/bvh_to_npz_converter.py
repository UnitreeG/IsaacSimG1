#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
BVH to NPZ converter for Isaac Lab motion imitation.

This script converts BVH (Biovision Hierarchy) motion capture files to NPZ format
compatible with Isaac Lab's motion imitation system.

Usage:
    python bvh_to_npz_converter.py --bvh_file path/to/macarena.bvh --output_file macarena_dance.npz --fps 30

Requirements:
    - bvh library (pip install bvh)
    - numpy
    - scipy for quaternion operations
"""

import argparse
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Optional

# Add support for BVH parsing
try:
    import bvh
except ImportError:
    print("BVH library not found. Please install it with: pip install bvh")
    sys.exit(1)

from scipy.spatial.transform import Rotation as R


class BVHToNPZConverter:
    """Converter class for BVH to NPZ format conversion."""
    
    def __init__(self, bvh_file: str, target_fps: int = 30):
        """Initialize the converter.
        
        Args:
            bvh_file: Path to the BVH file
            target_fps: Target FPS for the output motion data
        """
        self.bvh_file = bvh_file
        self.target_fps = target_fps
        
        # Load BVH file
        with open(bvh_file, 'r') as f:
            self.mocap = bvh.Bvh(f.read())
        
        # G1 robot joint mapping - map BVH joints to G1 robot joints
        self.g1_joint_mapping = {
            # Torso
            'torso_link': 'waist_yaw_joint',
            
            # Left leg
            'left_hip_yaw_link': 'left_hip_yaw_joint',
            'left_hip_roll_link': 'left_hip_roll_joint', 
            'left_hip_pitch_link': 'left_hip_pitch_joint',
            'left_knee_link': 'left_knee_joint',
            'left_ankle_pitch_link': 'left_ankle_pitch_joint',
            'left_ankle_roll_link': 'left_ankle_roll_joint',
            
            # Right leg
            'right_hip_yaw_link': 'right_hip_yaw_joint',
            'right_hip_roll_link': 'right_hip_roll_joint',
            'right_hip_pitch_link': 'right_hip_pitch_joint',
            'right_knee_link': 'right_knee_joint',
            'right_ankle_pitch_link': 'right_ankle_pitch_joint',
            'right_ankle_roll_link': 'right_ankle_roll_joint',
            
            # Left arm
            'left_shoulder_pitch_link': 'left_shoulder_pitch_joint',
            'left_shoulder_roll_link': 'left_shoulder_roll_joint',
            'left_shoulder_yaw_link': 'left_shoulder_yaw_joint',
            'left_elbow_link': 'left_elbow_joint',
            'left_wrist_roll_link': 'left_wrist_roll_joint',
            
            # Right arm
            'right_shoulder_pitch_link': 'right_shoulder_pitch_joint',
            'right_shoulder_roll_link': 'right_shoulder_roll_joint',
            'right_shoulder_yaw_link': 'right_shoulder_yaw_joint',
            'right_elbow_link': 'right_elbow_joint',
            'right_wrist_roll_link': 'right_wrist_roll_joint',
        }
        
        # G1 robot DOF names (23 DOF version)
        self.g1_dof_names = [
            'waist_yaw_joint',
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint', 'right_wrist_roll_joint'
        ]
        
        # G1 robot body names for tracking
        self.g1_body_names = [
            'base', 'torso_link',
            'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link',
            'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
            'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link',
            'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
            'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
            'left_elbow_link', 'left_wrist_roll_link',
            'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
            'right_elbow_link', 'right_wrist_roll_link'
        ]
        
        # BVH joint to G1 joint mapping for common mocap formats
        self.bvh_to_g1_joint_mapping = {
            # Common BVH joint names to G1 joints
            'Hips': 'base',
            'Spine': 'torso_link',
            'Spine1': 'torso_link',
            'Spine2': 'torso_link',
            'Neck': 'torso_link',
            'Head': 'torso_link',
            
            # Left leg
            'LeftUpLeg': 'left_hip_pitch_link',
            'LeftLeg': 'left_knee_link',
            'LeftFoot': 'left_ankle_pitch_link',
            'LeftToeBase': 'left_ankle_roll_link',
            
            # Right leg
            'RightUpLeg': 'right_hip_pitch_link',
            'RightLeg': 'right_knee_link',
            'RightFoot': 'right_ankle_pitch_link',
            'RightToeBase': 'right_ankle_roll_link',
            
            # Left arm
            'LeftShoulder': 'left_shoulder_pitch_link',
            'LeftArm': 'left_shoulder_roll_link',
            'LeftForeArm': 'left_elbow_link',
            'LeftHand': 'left_wrist_roll_link',
            
            # Right arm
            'RightShoulder': 'right_shoulder_pitch_link',
            'RightArm': 'right_shoulder_roll_link',
            'RightForeArm': 'right_elbow_link',
            'RightHand': 'right_wrist_roll_link',
        }
        
    def _extract_bvh_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract positions, rotations, and frame data from BVH file.
        
        Returns:
            Tuple of (positions, rotations, frame_times)
        """
        # Get frame data
        frame_count = self.mocap.nframes
        frame_time = self.mocap.frame_time
        original_fps = 1.0 / frame_time
        
        print(f"BVH file info:")
        print(f"  Frames: {frame_count}")
        print(f"  Frame time: {frame_time:.4f}s")
        print(f"  Original FPS: {original_fps:.1f}")
        print(f"  Target FPS: {self.target_fps}")
        
        # Get joint names from BVH
        joint_names = self.mocap.get_joints_names()
        print(f"  BVH joints: {joint_names}")
        
        # Resample to target FPS if needed
        if abs(original_fps - self.target_fps) > 0.1:
            frame_indices = np.linspace(0, frame_count - 1, 
                                      int(frame_count * self.target_fps / original_fps))
            frame_indices = np.round(frame_indices).astype(int)
            frame_indices = np.clip(frame_indices, 0, frame_count - 1)
        else:
            frame_indices = np.arange(frame_count)
        
        num_frames = len(frame_indices)
        
        # Extract frame data using the correct API
        positions = {}
        rotations = {}
        
        # Get the first frame to understand the data structure
        first_frame = self.mocap.frames[0]
        frame_values = [float(x) for x in first_frame]
        print(f"  Frame data length: {len(frame_values)}")
        
        # Root joint (Hips) has 6 DOF: position (XYZ) + rotation (XYZ)
        root_joint = joint_names[0]
        root_pos_data = []
        root_rot_data = []
        
        for frame_idx in frame_indices:
            frame_str_data = self.mocap.frames[frame_idx]
            frame_data = [float(x) for x in frame_str_data]
            
            # Root joint: position and rotation
            if len(frame_data) >= 6:
                pos = [frame_data[0], frame_data[1], frame_data[2]]  # X, Y, Z position
                rot = [frame_data[3], frame_data[4], frame_data[5]]  # X, Y, Z rotation
                root_pos_data.append(pos)
                root_rot_data.append(rot)
            else:
                root_pos_data.append([0.0, 0.0, 0.0])
                root_rot_data.append([0.0, 0.0, 0.0])
        
        # Store root joint data
        positions[root_joint] = np.array(root_pos_data)
        rotations[root_joint] = np.array(root_rot_data)
        
        # For other joints, extract 3DOF rotation data
        data_idx = 6  # Start after root joint's 6DOF
        
        for joint_name in joint_names[1:]:  # Skip root joint
            joint_rot_data = []
            
            for frame_idx in frame_indices:
                frame_str_data = self.mocap.frames[frame_idx]
                frame_data = [float(x) for x in frame_str_data]
                
                # Extract 3DOF rotation for this joint
                if len(frame_data) > data_idx + 2:
                    rot = [
                        frame_data[data_idx],
                        frame_data[data_idx + 1], 
                        frame_data[data_idx + 2]
                    ]
                    joint_rot_data.append(rot)
                else:
                    joint_rot_data.append([0.0, 0.0, 0.0])
            
            rotations[joint_name] = np.array(joint_rot_data)
            data_idx += 3  # Move to next joint's rotation data
        
        return positions, rotations, frame_indices
    
    def _convert_to_g1_format(self, positions: Dict, rotations: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert BVH data to G1 robot format.
        
        Args:
            positions: Dictionary of joint positions
            rotations: Dictionary of joint rotations
            
        Returns:
            Tuple of (dof_positions, dof_velocities, body_positions, body_rotations, body_lin_vel, body_ang_vel)
        """
        num_frames = len(list(positions.values())[0]) if positions else len(list(rotations.values())[0])
        
        # Initialize output arrays
        dof_positions = np.zeros((num_frames, len(self.g1_dof_names)))
        dof_velocities = np.zeros((num_frames, len(self.g1_dof_names)))
        body_positions = np.zeros((num_frames, len(self.g1_body_names), 3))
        body_rotations = np.zeros((num_frames, len(self.g1_body_names), 4))  # quaternions (w, x, y, z)
        body_lin_vel = np.zeros((num_frames, len(self.g1_body_names), 3))
        body_ang_vel = np.zeros((num_frames, len(self.g1_body_names), 3))
        
        # Convert BVH data to G1 format
        for frame_idx in range(num_frames):
            # Process each body
            for body_idx, body_name in enumerate(self.g1_body_names):
                # Find corresponding BVH joint
                bvh_joint = None
                for bvh_name, g1_name in self.bvh_to_g1_joint_mapping.items():
                    if g1_name == body_name:
                        bvh_joint = bvh_name
                        break
                
                if bvh_joint and bvh_joint in positions:
                    # Use BVH position data
                    pos = positions[bvh_joint][frame_idx]
                    body_positions[frame_idx, body_idx] = pos
                else:
                    # Use default position (could be improved with forward kinematics)
                    body_positions[frame_idx, body_idx] = [0.0, 0.0, 0.0]
                
                if bvh_joint and bvh_joint in rotations:
                    # Convert Euler angles to quaternions
                    euler = rotations[bvh_joint][frame_idx]
                    # Convert degrees to radians
                    euler_rad = np.radians(euler)
                    # Create rotation object (ZXY order is common in BVH)
                    rot = R.from_euler('zxy', euler_rad)
                    quat = rot.as_quat()  # returns [x, y, z, w]
                    # Convert to [w, x, y, z] format
                    body_rotations[frame_idx, body_idx] = [quat[3], quat[0], quat[1], quat[2]]
                else:
                    # Identity quaternion
                    body_rotations[frame_idx, body_idx] = [1.0, 0.0, 0.0, 0.0]
        
        # Calculate velocities using finite differences
        dt = 1.0 / self.target_fps
        
        for frame_idx in range(1, num_frames):
            # Linear velocities
            body_lin_vel[frame_idx] = (body_positions[frame_idx] - body_positions[frame_idx-1]) / dt
            
            # Angular velocities (simplified - could be improved)
            # For now, use finite differences on Euler angles
            for body_idx in range(len(self.g1_body_names)):
                q1 = body_rotations[frame_idx-1, body_idx]
                q2 = body_rotations[frame_idx, body_idx]
                
                # Convert quaternions to rotation objects
                r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])  # convert from [w,x,y,z] to [x,y,z,w]
                r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
                
                # Calculate relative rotation
                rel_rot = r2 * r1.inv()
                
                # Convert to angular velocity
                rotvec = rel_rot.as_rotvec()
                body_ang_vel[frame_idx, body_idx] = rotvec / dt
        
        # Convert body rotations to joint angles (simplified approach)
        # For now, we'll use a basic mapping - this could be improved with proper IK
        for frame_idx in range(num_frames):
            for dof_idx, dof_name in enumerate(self.g1_dof_names):
                # Simple mapping - could be improved with proper inverse kinematics
                if "hip_pitch" in dof_name:
                    dof_positions[frame_idx, dof_idx] = 0.0  # Default standing pose
                elif "knee" in dof_name:
                    dof_positions[frame_idx, dof_idx] = 0.0
                elif "ankle" in dof_name:
                    dof_positions[frame_idx, dof_idx] = 0.0
                elif "shoulder" in dof_name:
                    dof_positions[frame_idx, dof_idx] = 0.0
                elif "elbow" in dof_name:
                    dof_positions[frame_idx, dof_idx] = 0.0
                else:
                    dof_positions[frame_idx, dof_idx] = 0.0
        
        # Calculate DOF velocities
        for frame_idx in range(1, num_frames):
            dof_velocities[frame_idx] = (dof_positions[frame_idx] - dof_positions[frame_idx-1]) / dt
        
        return dof_positions, dof_velocities, body_positions, body_rotations, body_lin_vel, body_ang_vel
    
    def convert(self, output_file: str) -> None:
        """Convert BVH to NPZ format.
        
        Args:
            output_file: Output NPZ file path
        """
        print(f"Converting BVH file: {self.bvh_file}")
        print(f"Output file: {output_file}")
        
        # Extract BVH data
        positions, rotations, frame_indices = self._extract_bvh_data()
        
        # Convert to G1 format
        dof_positions, dof_velocities, body_positions, body_rotations, body_lin_vel, body_ang_vel = self._convert_to_g1_format(positions, rotations)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to NPZ format
        np.savez(
            output_file,
            fps=self.target_fps,
            dof_names=self.g1_dof_names,
            body_names=self.g1_body_names,
            dof_positions=dof_positions.astype(np.float32),
            dof_velocities=dof_velocities.astype(np.float32),
            body_positions=body_positions.astype(np.float32),
            body_rotations=body_rotations.astype(np.float32),
            body_linear_velocities=body_lin_vel.astype(np.float32),
            body_angular_velocities=body_ang_vel.astype(np.float32)
        )
        
        print(f"Conversion complete!")
        print(f"  Output frames: {len(dof_positions)}")
        print(f"  DOF count: {len(self.g1_dof_names)}")
        print(f"  Body count: {len(self.g1_body_names)}")
        print(f"  Duration: {len(dof_positions) / self.target_fps:.2f} seconds")


def main():
    """Main function for BVH to NPZ conversion."""
    parser = argparse.ArgumentParser(description="Convert BVH motion capture files to NPZ format for Isaac Lab")
    parser.add_argument("--bvh_file", type=str, required=True, help="Path to input BVH file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output NPZ file")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for output motion data")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.bvh_file):
        print(f"Error: BVH file not found: {args.bvh_file}")
        return
    
    # Create converter and convert
    converter = BVHToNPZConverter(args.bvh_file, args.fps)
    converter.convert(args.output_file)


if __name__ == "__main__":
    main() 