"""Unified Data Loader for Solar Sky Imagery and Pyranometer Irradiance.

Processes ground measurements, computes astronomical solar positions on the fly,
extracts theoretical clear-sky irradiance, and batches sequences of historical sky images.
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from ..physics.solar_geometry import SolarGeometry
from ..physics.clear_sky_models import ClearSkyModel, IneichenModel
from ..vision.image_preprocessing import SkyImagePreprocessor
from ..vision.circumsolar_extractor import CircumsolarExtractor


class SolarSkyDataset:
    """Dataset loader pairing historical sky image sequences with solar irradiance."""

    def __init__(
        self,
        images: np.ndarray,
        circumsolar_crops: np.ndarray,
        ghi_actual: np.ndarray,
        ghi_cs: np.ndarray,
        aux_features: np.ndarray,
        sequence_length: int = 4,
        forecast_steps_ahead: int = 1
    ):
        """
        Args:
            images: (N, H, W, 3) full-sky images.
            circumsolar_crops: (N, roi_H, roi_W, 3) sun-centered patches.
            ghi_actual: (N,) measured irradiance (W/m^2).
            ghi_cs: (N,) clear-sky irradiance (W/m^2).
            aux_features: (N, D) physical auxiliary variables.
            sequence_length: Number of past frames (e.g. 4 frames = [t-3, t-2, t-1, t]).
            forecast_steps_ahead: Target horizon (e.g. 1 step = +5 min ahead, 3 steps = +15 min ahead).
        """
        self.images = images
        self.circumsolar_crops = circumsolar_crops
        self.ghi_actual = ghi_actual
        self.ghi_cs = ghi_cs
        self.aux_features = aux_features
        self.seq_len = sequence_length
        self.horizon = forecast_steps_ahead

        # Valid index range
        self.num_samples = len(ghi_actual) - sequence_length - forecast_steps_ahead + 1

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # Sequence of past observations: [idx, ..., idx + seq_len - 1]
        t_current = idx + self.seq_len - 1
        t_target = t_current + self.horizon

        seq_images = self.images[idx : t_current + 1]
        seq_circ = self.circumsolar_crops[idx : t_current + 1]
        current_img = self.images[t_current]
        current_circ = self.circumsolar_crops[t_current]

        return {
            "seq_images": seq_images,
            "seq_circ": seq_circ,
            "current_image": current_img,
            "current_circ": current_circ,
            "aux_current": self.aux_features[t_current],
            "aux_target": self.aux_features[t_target],
            "ghi_current": self.ghi_actual[t_current],
            "ghi_cs_current": self.ghi_cs[t_current],
            "ghi_cs_target": self.ghi_cs[t_target],
            "ghi_target": self.ghi_actual[t_target],
        }

    def get_batch(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """Collects and stacks a batch of samples for fast vectorized training."""
        batch = [self[i] for i in indices]
        return {
            "seq_images": np.stack([b["seq_images"] for b in batch]),
            "current_image": np.stack([b["current_image"] for b in batch]),
            "current_circ": np.stack([b["current_circ"] for b in batch]),
            "aux_current": np.stack([b["aux_current"] for b in batch]),
            "aux_target": np.stack([b["aux_target"] for b in batch]),
            "ghi_current": np.array([b["ghi_current"] for b in batch], dtype=np.float32),
            "ghi_cs_current": np.array([b["ghi_cs_current"] for b in batch], dtype=np.float32),
            "ghi_cs_target": np.array([b["ghi_cs_target"] for b in batch], dtype=np.float32),
            "ghi_target": np.array([b["ghi_target"] for b in batch], dtype=np.float32),
        }
