"""Benchmark Runner: Side-by-side evaluation and visualization of all solar forecasting models."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from .metrics import SolarForecastingMetrics
from ..models.baselines import PersistenceModel, SmartPersistenceModel, ClearSkyOnlyModel, DirectRegressionBaseline
from ..models.residual_network import GaoLiuResidualNetwork
from ..models.novel_sun_attention_net import DualStreamCircumsolarResidualNet


class BenchmarkRunner:
    """Executes multi-model comparisons and renders evaluation plots."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_benchmark(
        self,
        test_data: Dict[str, np.ndarray],
        gao_liu_model: GaoLiuResidualNetwork,
        novel_model: DualStreamCircumsolarResidualNet
    ) -> Dict[str, Dict[str, float]]:
        """Runs side-by-side benchmarking across all models.

        Args:
            test_data: Dictionary containing test images, crops, actual GHI, clear sky GHI, and aux features.
            gao_liu_model: Trained Gao & Liu (WACV 2022) baseline.
            novel_model: Trained DS-CARN novel model.

        Returns:
            Dictionary of metrics for each model.
        """
        images = test_data["images"]
        crops = test_data["circumsolar_crops"]
        ghi_actual = test_data["ghi_actual"]
        ghi_cs = test_data["ghi_cs"]
        aux_features = test_data["aux_features"]
        n = len(ghi_actual)

        # Shifted targets for +1 step ahead forecast (nowcasting or 5-15 min horizon)
        y_prev = ghi_actual[:-1]
        y_true = ghi_actual[1:]
        cs_target = ghi_cs[1:]
        cs_current = ghi_cs[:-1]
        img_current = images[:-1]
        crop_current = crops[:-1]
        aux_current = aux_features[:-1]

        # 1. Persistence Baseline
        pers_model = PersistenceModel()
        pred_persistence = pers_model.predict(y_prev)

        # 2. Clear Sky Only Baseline
        cs_model = ClearSkyOnlyModel()
        pred_clearsky = cs_model.predict(cs_target)

        # 3. Smart Persistence (Clear-Sky Index) - standard literature reference
        smart_model = SmartPersistenceModel()
        pred_smart = smart_model.predict(y_prev, cs_current, cs_target)

        # 4. Direct Vision Regression (No Clear Sky Model)
        direct_model = DirectRegressionBaseline(feature_dim=64+5)
        # Quick fit on earlier portion
        direct_feats = np.concatenate([
            gao_liu_model.extract_image_features(img_current),
            aux_current
        ], axis=-1)
        pred_direct = direct_model.predict(
            gao_liu_model.extract_image_features(img_current),
            aux_current
        )

        # 5. Gao & Liu (WACV 2022) Clear-Sky Residual Model
        pred_gao_liu = gao_liu_model.predict(img_current, aux_current, cs_target)

        # 6. Novel Dual-Stream Circumsolar-Aware Residual Net (DS-CARN)
        novel_res = novel_model.predict(img_current, crop_current, aux_current, cs_target)
        pred_novel = novel_res["ghi_pred"]
        pred_p10 = novel_res["ghi_p10"]
        pred_p90 = novel_res["ghi_p90"]

        predictions = {
            "Persistence": pred_persistence,
            "Clear-Sky Only": pred_clearsky,
            "Smart Persistence (k_t)": pred_smart,
            "Direct Vision Regression (No CSM)": pred_direct,
            "Gao & Liu (WACV 2022) Residual": pred_gao_liu,
            "Novel DS-CARN (Ours)": pred_novel
        }

        # Calculate metrics against Smart Persistence baseline
        results = {}
        for name, preds in predictions.items():
            results[name] = SolarForecastingMetrics.evaluate_all(
                y_true=y_true,
                y_pred=preds,
                baseline_pred=pred_smart,
                y_prev=y_prev
            )

        # Save metrics JSON
        metrics_file = os.path.join(self.output_dir, "benchmark_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2)

        # Generate visual plots
        self._plot_timeseries_comparison(
            y_true=y_true,
            ghi_cs=cs_target,
            predictions=predictions,
            p10=pred_p10,
            p90=pred_p90
        )
        self._plot_metrics_bar_chart(results)

        return results

    def _plot_timeseries_comparison(
        self,
        y_true: np.ndarray,
        ghi_cs: np.ndarray,
        predictions: Dict[str, np.ndarray],
        p10: np.ndarray,
        p90: np.ndarray
    ):
        """Generates publication-quality irradiance time series forecasting plot."""
        fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
        time_steps = np.arange(len(y_true))

        # Clear-sky envelope
        ax.plot(time_steps, ghi_cs, label="Theoretical Clear-Sky (Ineichen CSM)", color="orange", linestyle="--", alpha=0.7, linewidth=1.5)
        # Ground Truth
        ax.plot(time_steps, y_true, label="Ground Truth Pyranometer GHI", color="black", linewidth=2.2)

        # Key Models
        ax.plot(time_steps, predictions["Smart Persistence (k_t)"], label="Smart Persistence Baseline", color="gray", linestyle=":", alpha=0.8)
        ax.plot(time_steps, predictions["Gao & Liu (WACV 2022) Residual"], label="Gao & Liu (WACV 2022)", color="blue", linewidth=1.8, alpha=0.85)
        ax.plot(time_steps, predictions["Novel DS-CARN (Ours)"], label="Novel DS-CARN (Circumsolar-Aware)", color="crimson", linewidth=2.0)

        # Uncertainty bounds for novel model
        ax.fill_between(time_steps, p10, p90, color="crimson", alpha=0.18, label="DS-CARN 80% Prediction Interval [P10, P90]")

        ax.set_title("Short-Term Solar Irradiance Forecasting: Model Comparison under Cloud Transients", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time Step (5-min intervals)", fontsize=12)
        ax.set_ylabel("Global Horizontal Irradiance GHI ($W/m^2$)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

        save_path = os.path.join(self.output_dir, "timeseries_forecast.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved time-series forecast plot to {save_path}")

    def _plot_metrics_bar_chart(self, results: Dict[str, Dict[str, float]]):
        """Generates RMSE and Forecast Skill Score comparison bar charts."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        models = list(results.keys())
        rmse_vals = [results[m]["RMSE"] for m in models]
        skill_vals = [results[m].get("Skill_Score_%", 0.0) for m in models]

        colors = ["#7f7f7f", "#bcbd22", "#17becf", "#9467bd", "#1f77b4", "#d62728"]

        # Bar chart 1: RMSE (Lower is better)
        bars1 = ax1.barh(models, rmse_vals, color=colors)
        ax1.set_xlabel("RMSE ($W/m^2$) [Lower is Better]", fontsize=11, fontweight="bold")
        ax1.set_title("Forecast Accuracy (RMSE)", fontsize=13, fontweight="bold")
        ax1.grid(True, axis="x", linestyle="--", alpha=0.4)
        for bar in bars1:
            w = bar.get_width()
            ax1.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}", va="center", fontsize=9.5)

        # Bar chart 2: Forecast Skill Score vs Smart Persistence (Higher is better)
        bars2 = ax2.barh(models, skill_vals, color=colors)
        ax2.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax2.set_xlabel("Forecast Skill Score % [Higher is Better]", fontsize=11, fontweight="bold")
        ax2.set_title("Forecast Skill Score ($SS_{RMSE}$)", fontsize=13, fontweight="bold")
        ax2.grid(True, axis="x", linestyle="--", alpha=0.4)
        for bar in bars2:
            w = bar.get_width()
            offset = 1.0 if w >= 0 else -6.0
            ax2.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9.5)

        save_path = os.path.join(self.output_dir, "benchmark_comparison.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved benchmark comparison chart to {save_path}")
