"""Baseline Forecasting Models for Short-term Solar Irradiance.

Includes:
1. Standard Persistence: GHI(t + dt) = GHI(t).
2. Smart Persistence (Clear-Sky Index Persistence): k_t(t + dt) = k_t(t) => GHI(t + dt) = k_t * GHI_cs(t + dt).
3. Pure Clear-Sky Model: GHI(t + dt) = GHI_cs(t + dt).
4. Direct Vision Regression (no physics): f(Image) -> GHI directly.
"""

import numpy as np
from typing import Union, Tuple, Optional


class PersistenceModel:
    """Standard Persistence Baseline.

    Assumes future solar irradiance remains identical to current observation.
    GHI(t + horizon) = GHI(t).
    """

    def predict(self, current_ghi: np.ndarray) -> np.ndarray:
        return np.asarray(current_ghi, dtype=np.float32)


class SmartPersistenceModel:
    """Smart Persistence Baseline (Clear-Sky Index Persistence).

    The standard peer-reviewed benchmark. Holds the clear-sky index k_t constant
    while adjusting for the astronomical progression of solar geometry.
    GHI(t + dt) = (GHI(t) / GHI_cs(t)) * GHI_cs(t + dt).
    """

    def predict(
        self,
        current_ghi: np.ndarray,
        current_ghi_cs: np.ndarray,
        future_ghi_cs: np.ndarray,
        epsilon: float = 1e-4
    ) -> np.ndarray:
        current_ghi = np.asarray(current_ghi, dtype=np.float32)
        current_ghi_cs = np.asarray(current_ghi_cs, dtype=np.float32)
        future_ghi_cs = np.asarray(future_ghi_cs, dtype=np.float32)

        # Clear sky index at observation time
        kt = current_ghi / np.maximum(current_ghi_cs, epsilon)
        # Apply to future clear sky irradiance
        future_ghi = kt * future_ghi_cs
        return np.maximum(0.0, future_ghi)


class ClearSkyOnlyModel:
    """Clear-Sky Baseline.

    Assumes 100% clear sky conditions at the target forecast horizon.
    GHI(t + dt) = GHI_cs(t + dt).
    """

    def predict(self, future_ghi_cs: np.ndarray) -> np.ndarray:
        return np.asarray(future_ghi_cs, dtype=np.float32)


class DirectRegressionBaseline:
    """Direct end-to-end regression baseline without clear sky model guidance.

    Demonstrates the limitation of forcing a vision model to learn both
    celestial diurnal curves and cloud dynamics simultaneously.
    """

    def __init__(self, feature_dim: int = 64, lr: float = 0.01):
        self.weights = np.random.randn(feature_dim, 1) * 0.05
        self.bias = np.array([400.0]) # starting around mid-day solar average

    def predict(self, image_features: np.ndarray, aux_features: np.ndarray) -> np.ndarray:
        combined = np.concatenate([image_features, aux_features], axis=-1)
        pred = np.dot(combined, self.weights) + self.bias
        return np.maximum(0.0, pred.squeeze())

    def fit_step(self, x_features: np.ndarray, y_true: np.ndarray, lr: float = 0.001):
        pred = np.dot(x_features, self.weights) + self.bias
        grad_w = np.dot(x_features.T, (pred - y_true[:, None])) / len(x_features)
        grad_b = np.mean(pred - y_true[:, None], axis=0)
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b
