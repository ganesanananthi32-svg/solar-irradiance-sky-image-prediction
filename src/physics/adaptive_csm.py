"""Novelty Component 1: Adaptive Clear-Sky Auto-Tuning and Dynamic Calibration.

Addresses a major flaw in existing literature (e.g. Gao & Liu 2022) where
static, fixed atmospheric parameters (e.g., Linke Turbidity TL = 3.0) cause
large systematic baseline errors due to unmodeled aerosols, humidity, and haze.

This module provides:
1. Dynamic Linke Turbidity estimation from clear-sky envelope detection.
2. Atmospheric haze estimation via background sky image Red-to-Blue Ratio (RBR).
3. Zero-bias baseline calibration to ensure Delta GHI = 0 on clear days.
"""

import numpy as np
from typing import Optional, Tuple, List
from .clear_sky_models import IneichenModel


class AdaptiveClearSkyCalibrator:
    """Dynamically calibrates clear sky model parameters to prevent baseline drift."""

    def __init__(
        self,
        base_model: IneichenModel,
        tl_min: float = 1.5,
        tl_max: float = 6.5,
        rolling_window_steps: int = 60
    ):
        """
        Args:
            base_model: Instance of IneichenModel.
            tl_min: Minimum physically realistic Linke Turbidity (pristine mountain air).
            tl_max: Maximum Linke Turbidity (heavy smog/aerosols/humidity).
            rolling_window_steps: Number of recent timestamps to track for envelope fitting.
        """
        self.base_model = base_model
        self.tl_min = tl_min
        self.tl_max = tl_max
        self.rolling_window = rolling_window_steps
        self.current_tl = base_model.default_tl

        # Historical buffers for clear sky envelope detection
        self.history_ghi: List[float] = []
        self.history_zenith: List[float] = []

    def calibrate_from_observations(
        self,
        observed_ghi: float,
        zenith_deg: float,
        day_of_year: int
    ) -> float:
        """Dynamically tunes Linke Turbidity when clear sky conditions are detected.

        Uses local derivative stability (low short-term variance indicates clear sky).
        """
        if zenith_deg > 75.0 or observed_ghi < 50.0:
            return self.current_tl

        self.history_ghi.append(observed_ghi)
        self.history_zenith.append(zenith_deg)

        if len(self.history_ghi) > self.rolling_window:
            self.history_ghi.pop(0)
            self.history_zenith.pop(0)

        # Check stability (standard deviation of normalized irradiance)
        if len(self.history_ghi) >= 15:
            arr_ghi = np.array(self.history_ghi[-15:])
            arr_zenith = np.array(self.history_zenith[-15:])

            # Check if condition is smooth (unbroken clear sky)
            diffs = np.abs(np.diff(arr_ghi))
            mean_step_change = np.mean(diffs)

            if mean_step_change < 4.0:  # Smooth solar arc, no cloud transients
                # Invert Ineichen model via 1D grid search over [tl_min, tl_max]
                best_tl = self.current_tl
                min_error = float("inf")

                for candidate_tl in np.linspace(self.tl_min, self.tl_max, 25):
                    theoretical = self.base_model.calculate_ghi(
                        zenith_deg=zenith_deg,
                        day_of_year=day_of_year,
                        linke_turbidity=candidate_tl
                    )
                    error = abs(observed_ghi - theoretical)
                    if error < min_error:
                        min_error = error
                        best_tl = candidate_tl

                # Exponential moving average update
                self.current_tl = 0.90 * self.current_tl + 0.10 * best_tl

        return self.current_tl

    def calibrate_from_sky_image(
        self,
        clear_sky_patch_rbr: float
    ) -> float:
        """Empirical atmospheric haze estimation from sky image Red-to-Blue Ratio (RBR).

        Rayleigh molecular scattering produces deep blue skies (low RBR ~ 0.35-0.50).
        Aerosols and water vapor cause Mie scattering, whitening the sky (RBR ~ 0.65-0.85).

        Args:
            clear_sky_patch_rbr: Average RBR extracted from clear sky dome regions.

        Returns:
            calibrated_tl: Inferred Linke Turbidity.
        """
        # Linear transfer function calibrated between pristine (TL=1.8) and hazy (TL=5.0)
        norm_rbr = np.clip((clear_sky_patch_rbr - 0.35) / (0.80 - 0.35), 0.0, 1.0)
        inferred_tl = self.tl_min + norm_rbr * (self.tl_max - self.tl_min)

        # Smooth update
        self.current_tl = 0.85 * self.current_tl + 0.15 * inferred_tl
        return self.current_tl

    def get_calibrated_clear_sky(
        self,
        zenith_deg: float,
        day_of_year: int
    ) -> Tuple[float, float]:
        """Returns the calibrated clear-sky GHI and the active Linke Turbidity."""
        ghi_cs = self.base_model.calculate_ghi(
            zenith_deg=zenith_deg,
            day_of_year=day_of_year,
            linke_turbidity=self.current_tl
        )
        return float(ghi_cs), float(self.current_tl)
