"""Novelty Component 3: Physics-Constrained Ramp-Aware Loss and Quantile Pinball Loss.

Addresses the critical failure mode of standard MSE/MAE loss functions:
1. MSE creates overly smooth predictions that miss rapid cloud ramps (dGHI/dt),
   which are catastrophic for power grid stability and inverter operations.
2. Standard losses ignore physical boundaries, sometimes predicting negative
   irradiance or values violating atmospheric upper limits.

This module provides:
1. PhysicsRampLoss: Jointly optimizes irradiance error, ramp rate error, and physical bounds.
2. QuantilePinballLoss: Enables probabilistic solar nowcasting with prediction intervals.
"""

import numpy as np
from typing import Optional, Dict


class PhysicsRampLoss:
    """Physics-constrained loss penalizing ramp errors and non-physical values."""

    def __init__(
        self,
        ramp_weight: float = 0.35,
        bound_weight: float = 0.50,
        max_clear_sky_enhancement_factor: float = 1.25
    ):
        """
        Args:
            ramp_weight: Weight lambda for the ramp rate discrepancy penalty.
            bound_weight: Weight penalty for violating physical boundaries [0, 1.25 * GHI_cs].
            max_clear_sky_enhancement_factor: Maximum physical cloud enhancement multiplier.
        """
        self.ramp_weight = ramp_weight
        self.bound_weight = bound_weight
        self.max_enhancement = max_clear_sky_enhancement_factor

    def __call__(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        y_prev: Optional[np.ndarray] = None,
        ghi_cs: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculates loss and component breakdowns.

        Args:
            y_pred: Predicted GHI (N,).
            y_true: Ground truth GHI (N,).
            y_prev: Previous observed GHI at t-1 (N,) for ramp calculation.
            ghi_cs: Clear sky GHI at forecast horizon (N,).

        Returns:
            Dictionary with 'total_loss', 'mse_loss', 'ramp_loss', 'bound_loss'.
        """
        y_pred = np.asarray(y_pred, dtype=np.float32)
        y_true = np.asarray(y_true, dtype=np.float32)

        # 1. Base Mean Squared Error
        mse_loss = float(np.mean((y_pred - y_true) ** 2))

        # 2. Ramp Rate Loss: d(GHI)/dt
        ramp_loss = 0.0
        if y_prev is not None:
            pred_ramp = y_pred - y_prev
            true_ramp = y_true - y_prev
            ramp_loss = float(np.mean((pred_ramp - true_ramp) ** 2))

        # 3. Physical Bound Violations:
        # Constraint A: GHI >= 0 (no negative solar radiation)
        # Constraint B: GHI <= 1.25 * GHI_cs (physical ceiling for circumsolar cloud lensing)
        neg_penalty = np.mean(np.maximum(0.0, -y_pred) ** 2)
        bound_loss = float(neg_penalty)

        if ghi_cs is not None:
            max_limit = ghi_cs * self.max_enhancement
            upper_penalty = np.mean(np.maximum(0.0, y_pred - max_limit) ** 2)
            bound_loss += float(upper_penalty)

        total_loss = mse_loss + (self.ramp_weight * ramp_loss) + (self.bound_weight * bound_loss)

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "ramp_loss": ramp_loss,
            "bound_loss": bound_loss
        }


class QuantilePinballLoss:
    """Asymmetric pinball loss for probabilistic solar irradiance prediction intervals."""

    def __init__(self, quantiles=(0.10, 0.50, 0.90)):
        self.quantiles = quantiles

    def __call__(
        self,
        y_preds: Dict[float, np.ndarray],
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Args:
            y_preds: Dictionary mapping quantile tau to predicted array (N,).
            y_true: Ground truth GHI array (N,).

        Returns:
            Dictionary of pinball losses per quantile and overall mean loss.
        """
        total_loss = 0.0
        details = {}
        for tau in self.quantiles:
            pred = y_preds[tau]
            err = y_true - pred
            loss_tau = float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))
            details[f"pinball_q{int(tau * 100)}"] = loss_tau
            total_loss += loss_tau

        details["mean_pinball_loss"] = total_loss / len(self.quantiles)
        return details
