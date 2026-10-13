"""Evaluation Metrics for Solar Irradiance Forecasting.

Implements academic and industry benchmark metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MBE (Mean Bias Error)
- rRMSE (Normalized/Relative RMSE percentage)
- Forecast Skill Score (SS_RMSE) relative to Smart Persistence
- Ramp Detection F1-Score (performance on steep cloud occlusion events)
"""

import numpy as np
from typing import Dict, Any, Optional


class SolarForecastingMetrics:
    """Computes comprehensive deterministic and ramp evaluation metrics."""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error in W/m^2."""
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error in W/m^2."""
        return float(np.mean(np.abs(y_pred - y_true)))

    @staticmethod
    def mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Bias Error in W/m^2 (positive = overprediction, negative = underprediction)."""
        return float(np.mean(y_pred - y_true))

    @staticmethod
    def r_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Relative RMSE percentage (%)."""
        mean_true = np.mean(y_true)
        if mean_true <= 1e-3:
            return 0.0
        return float((SolarForecastingMetrics.rmse(y_true, y_pred) / mean_true) * 100.0)

    @staticmethod
    def forecast_skill_score(
        rmse_model: float,
        rmse_baseline: float
    ) -> float:
        """Forecast Skill Score percentage relative to Smart Persistence.

        SS = (1 - RMSE_model / RMSE_smart_persistence) * 100%
        Positive values indicate superior performance over standard reference.
        """
        if rmse_baseline <= 1e-4:
            return 0.0
        return float((1.0 - (rmse_model / rmse_baseline)) * 100.0)

    @staticmethod
    def ramp_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prev: np.ndarray,
        ramp_threshold_w_m2: float = 80.0
    ) -> Dict[str, float]:
        """Evaluates detection of severe solar ramp events (|Delta GHI| > threshold).

        Returns:
            precision, recall, f1_score of ramp detection.
        """
        true_ramps = np.abs(y_true - y_prev) >= ramp_threshold_w_m2
        pred_ramps = np.abs(y_pred - y_prev) >= ramp_threshold_w_m2

        tp = np.count_nonzero(true_ramps & pred_ramps)
        fp = np.count_nonzero((~true_ramps) & pred_ramps)
        fn = np.count_nonzero(true_ramps & (~pred_ramps))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "ramp_precision": precision,
            "ramp_recall": recall,
            "ramp_f1": f1,
            "total_ramps_observed": int(np.count_nonzero(true_ramps)),
            "total_ramps_predicted": int(np.count_nonzero(pred_ramps))
        }

    @classmethod
    def evaluate_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_pred: Optional[np.ndarray] = None,
        y_prev: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Computes full suite of evaluation metrics."""
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)

        metrics = {
            "RMSE": cls.rmse(y_true, y_pred),
            "MAE": cls.mae(y_true, y_pred),
            "MBE": cls.mbe(y_true, y_pred),
            "rRMSE_%": cls.r_rmse(y_true, y_pred)
        }

        if baseline_pred is not None:
            baseline_rmse = cls.rmse(y_true, baseline_pred)
            metrics["Baseline_RMSE"] = baseline_rmse
            metrics["Skill_Score_%"] = cls.forecast_skill_score(metrics["RMSE"], baseline_rmse)

        if y_prev is not None:
            ramp_res = cls.ramp_score(y_true, y_pred, y_prev)
            metrics.update(ramp_res)

        return metrics
