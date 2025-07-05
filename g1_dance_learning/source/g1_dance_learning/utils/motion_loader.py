# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion loader utility for G1 dance learning."""

import numpy as np
import os
import torch
from typing import Dict, Optional


class MotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, device: torch.device) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)

        self.device = device
        self._dof_names = data["dof_names"].tolist()
        self._body_names = data["body_names"].tolist()

        self.dof_positions = torch.tensor(data["dof_positions"], dtype=torch.float32, device=self.device)
        self.dof_velocities = torch.tensor(data["dof_velocities"], dtype=torch.float32, device=self.device)
        self.body_positions = torch.tensor(data["body_positions"], dtype=torch.float32, device=self.device)
        self.body_rotations = torch.tensor(data["body_rotations"], dtype=torch.float32, device=self.device)
        self.body_linear_velocities = torch.tensor(
            data["body_linear_velocities"], dtype=torch.float32, device=self.device
        )
        self.body_angular_velocities = torch.tensor(
            data["body_angular_velocities"], dtype=torch.float32, device=self.device
        )

        self.dt = 1.0 / data["fps"]
        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.dt * (self.num_frames - 1)
        print(f"Motion loaded ({motion_file}): duration: {self.duration} sec, frames: {self.num_frames}")

    @property
    def num_dofs(self) -> int:
        """Number of degrees of freedom."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of bodies."""
        return len(self._body_names)

    @property
    def dof_names(self) -> list:
        """DOF names."""
        return self._dof_names.copy()

    @property
    def body_names(self) -> list:
        """Body names."""
        return self._body_names.copy()

    def get_dof_index(self, dof_names: list) -> list:
        """Get DOF indices for given DOF names.

        Args:
            dof_names: List of DOF names to query.

        Returns:
            List of DOF indices corresponding to the provided DOF names.
        """
        dof_indices = []
        for dof_name in dof_names:
            if dof_name in self._dof_names:
                dof_indices.append(self._dof_names.index(dof_name))
        return dof_indices

    def get_body_index(self, body_names: list) -> list:
        """Get body indices for given body names.

        Args:
            body_names: List of body names to query.

        Returns:
            List of body indices corresponding to the provided body names.
        """
        body_indices = []
        for body_name in body_names:
            if body_name in self._body_names:
                body_indices.append(self._body_names.index(body_name))
        return body_indices

    def sample_times(self, num_samples: int, duration: float | None = None) -> np.ndarray:
        """Sample random times within the motion duration.

        Args:
            num_samples: Number of time samples to generate.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.

        Returns:
            Sampled motion times.
        """
        if duration is None:
            duration = self.duration
        else:
            duration = min(duration, self.duration)

        times = np.random.uniform(0.0, duration, size=num_samples)
        return times

    def sample(
        self, num_samples: int, times: Optional[np.ndarray] = None, duration: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            Sampled motion DOF positions (with shape (N, num_dofs)), DOF velocities (with shape (N, num_dofs)),
            body positions (with shape (N, num_bodies, 3)), body rotations (with shape (N, num_bodies, 4), as wxyz quaternion),
            body linear velocities (with shape (N, num_bodies, 3)) and body angular velocities (with shape (N, num_bodies, 3)).
        """
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(self.dof_positions, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_positions, blend=blend, start=index_0, end=index_1),
            self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_linear_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_angular_velocities, blend=blend, start=index_0, end=index_1),
        )

    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute frame blending parameters for given times.

        Args:
            times: Array of time values.

        Returns:
            Tuple of (start_indices, end_indices, blend_weights).
        """
        frame_times = times / self.dt
        frame_indices = np.floor(frame_times).astype(int)
        frame_indices = np.clip(frame_indices, 0, self.num_frames - 2)

        blend = frame_times - frame_indices
        blend = np.clip(blend, 0.0, 1.0)

        return frame_indices, frame_indices + 1, blend

    def _interpolate(self, data: torch.Tensor, blend: torch.Tensor, start: np.ndarray, end: np.ndarray) -> torch.Tensor:
        """Linear interpolation between two data points.

        Args:
            data: Data tensor to interpolate.
            blend: Blending weights.
            start: Start indices.
            end: End indices.

        Returns:
            Interpolated data.
        """
        data_start = data[start]
        data_end = data[end]
        
        # Reshape blend to match data dimensions
        blend_expanded = blend.view(-1, *[1] * (data_start.dim() - 1))
        
        return (1.0 - blend_expanded) * data_start + blend_expanded * data_end

    def _slerp(self, quaternions: torch.Tensor, blend: torch.Tensor, start: np.ndarray, end: np.ndarray) -> torch.Tensor:
        """Spherical linear interpolation for quaternions.

        Args:
            quaternions: Quaternion data tensor.
            blend: Blending weights.
            start: Start indices.
            end: End indices.

        Returns:
            Interpolated quaternions.
        """
        # For simplicity, use linear interpolation and normalize
        # In a production system, you'd want proper SLERP
        q_start = quaternions[start]
        q_end = quaternions[end]
        
        # Reshape blend to match quaternion dimensions
        blend_expanded = blend.view(-1, *[1] * (q_start.dim() - 1))
        
        # Linear interpolation
        result = (1.0 - blend_expanded) * q_start + blend_expanded * q_end
        
        # Normalize quaternions
        result = result / torch.norm(result, dim=-1, keepdim=True)
        
        return result


# Motion data cache for efficient loading
_motion_cache = {}


def load_motion_data(motion_file: str, device: torch.device) -> MotionLoader:
    """Load motion data from NPZ file with caching.
    
    Args:
        motion_file: Path to the motion NPZ file
        device: Device to load the data on
        
    Returns:
        MotionLoader instance
    """
    cache_key = (motion_file, str(device))
    
    if cache_key not in _motion_cache:
        _motion_cache[cache_key] = MotionLoader(motion_file, device)
    
    return _motion_cache[cache_key] 