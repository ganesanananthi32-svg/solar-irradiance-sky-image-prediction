"""End-to-End Synthetic Benchmark Script.

Executes a complete training and benchmarking run comparing:
1. Persistence
2. Clear-Sky Only (Ineichen Model)
3. Smart Persistence (Clear-Sky Index Baseline)
4. Direct Vision Regression (No Clear-Sky Model)
5. Gao & Liu (WACV 2022) Clear-Sky Residual Network
6. Our Novel DS-CARN (Dual-Stream Circumsolar-Aware Residual Net)

Generates evaluation charts and metrics summary in `results/`.
"""

import os
import sys
import datetime
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic_benchmarks import SyntheticSolarBenchmarkGenerator
from src.models.residual_network import GaoLiuResidualNetwork
from src.models.novel_sun_attention_net import DualStreamCircumsolarResidualNet
from src.evaluation.benchmark_runner import BenchmarkRunner


def main():
    print("=" * 80)
    print("SHORT-TERM SOLAR IRRADIANCE FORECASTING BENCHMARK")
    print("TOPIC: Sky Images with Clear-Sky Models")
    print("=" * 80)

    # 1. Generate Synthetic Data
    print("\n[Step 1/4] Generating Physics-Grounded Solar Dataset...")
    generator = SyntheticSolarBenchmarkGenerator(image_size=128, seed=42)

    # Generate training day (broken clouds)
    train_data = generator.generate_day_sequence(
        day_date=datetime.date(2024, 6, 21),
        start_hour=8,
        end_hour=17,
        step_minutes=5,
        weather_type="broken_clouds"
    )
    print(f"  Generated {len(train_data['ghi_actual'])} training timesteps.")

    # Generate testing day (ramp storm: rapid cloud transitions)
    test_data = generator.generate_day_sequence(
        day_date=datetime.date(2024, 6, 22),
        start_hour=8,
        end_hour=17,
        step_minutes=5,
        weather_type="ramp_storm"
    )
    print(f"  Generated {len(test_data['ghi_actual'])} testing timesteps with dynamic solar ramps.")

    # 2. Initialize Models
    print("\n[Step 2/4] Initializing Deep Learning Models...")
    gao_liu_model = GaoLiuResidualNetwork(image_feature_dim=64, aux_dim=5, hidden_dim=128, seed=42)
    novel_model = DualStreamCircumsolarResidualNet(global_dim=32, circumsolar_dim=32, aux_dim=5, fusion_dim=64, seed=42)
    print("  Initialized Gao & Liu (WACV 2022) Clear-Sky Residual Network.")
    print("  Initialized Novel DS-CARN (Dual-Stream Circumsolar-Aware Residual Network).")

    # 3. Train Models
    print("\n[Step 3/4] Training Models on Solar Irradiance Residuals...")
    tr_images = train_data["images"][:-1]
    tr_crops = train_data["circumsolar_crops"][:-1]
    tr_aux = train_data["aux_features"][:-1]
    tr_cs_target = train_data["ghi_cs"][1:]
    tr_target = train_data["ghi_actual"][1:]
    tr_prev = train_data["ghi_actual"][:-1]

    epochs = 40
    print(f"  Training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        loss_gl = gao_liu_model.train_step(
            images=tr_images,
            aux_features=tr_aux,
            ghi_cs=tr_cs_target,
            ghi_target=tr_target,
            lr=0.003
        )
        loss_novel = novel_model.train_step_ramp(
            full_images=tr_images,
            circumsolar_crops=tr_crops,
            aux_features=tr_aux,
            ghi_cs=tr_cs_target,
            ghi_target=tr_target,
            ghi_prev=tr_prev,
            ramp_weight=0.40,
            lr=0.003
        )
        if epoch % 10 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:2d}/{epochs:2d} | Gao&Liu Loss: {loss_gl:7.2f} | DS-CARN Loss: {loss_novel:7.2f}")

    # 4. Benchmark & Evaluate
    print("\n[Step 4/4] Running Comprehensive Multi-Model Benchmark on Test Set...")
    runner = BenchmarkRunner(output_dir="results")
    results = runner.run_benchmark(test_data, gao_liu_model, novel_model)

    # Print Formatted Results Table
    print("\n" + "=" * 90)
    print(f"{'Model':<35} | {'RMSE (W/m2)':<12} | {'MAE (W/m2)':<10} | {'rRMSE (%)':<10} | {'Skill Score (%)':<15}")
    print("-" * 90)
    for model_name, m in results.items():
        skill_str = f"{m.get('Skill_Score_%', 0.0):+.2f}%" if "Skill_Score_%" in m else "REF"
        print(f"{model_name:<35} | {m['RMSE']:<12.2f} | {m['MAE']:<10.2f} | {m['rRMSE_%']:<10.2f} | {skill_str:<15}")
    print("=" * 90)

    print("\nBenchmark completed successfully!")
    print("Artifacts generated:")
    print("  - Results plot: results/timeseries_forecast.png")
    print("  - Benchmark comparison: results/benchmark_comparison.png")
    print("  - Metrics JSON: results/benchmark_metrics.json")


if __name__ == "__main__":
    main()
