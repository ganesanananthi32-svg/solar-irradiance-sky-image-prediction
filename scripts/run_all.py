"""Run All: Master Execution Script for Real Solar Forecasting Pipeline.

Runs Steps 1, 2, and 3 sequentially:
- Step 1: Downloads & builds real Stanford SKIPP'D sky camera dataset.
- Step 2: Visualizes sun tracking, circumsolar ROI, and log-polar warping.
- Step 3: Trains & benchmarks models on real sky data with conformal bounds.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scripts.step1_download_real_data as step1
import scripts.step2_visualize_real_sun_tracking as step2
import scripts.step3_train_and_benchmark_real_data as step3


def main():
    print("*" * 80)
    print("MASTER EXECUTION: REAL SOLAR IRRADIANCE & SKY IMAGE BENCHMARK")
    print("*" * 80)

    print("\n>>> EXECUTING STEP 1...")
    step1.main()

    print("\n>>> EXECUTING STEP 2...")
    step2.main()

    print("\n>>> EXECUTING STEP 3...")
    step3.main()

    print("\n" + "*" * 80)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("Inspect your results in `results/real_data/`:")
    print("  1. results/real_data/real_sun_polar_warp_sample.png")
    print("  2. results/real_data/real_timeseries_forecast.png")
    print("  3. results/real_data/real_benchmark_comparison.png")
    print("  4. results/real_data/real_benchmark_metrics.json")
    print("*" * 80)


if __name__ == "__main__":
    main()
