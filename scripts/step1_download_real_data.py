"""Step 1: Download & Build Real Stanford SKIPP'D All-Sky Dataset.

Acquires authentic ground-based all-sky fish-eye camera frames,
calculates solar positions, and compiles `data/real_dataset_manifest.csv`.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.real_data_pipeline import RealSkyDataPipeline


def main():
    print("=" * 80)
    print("STEP 1: ACQUIRING REAL ALL-SKY CAMERA DATASET (STANFORD SKIPP'D)")
    print("=" * 80)

    pipeline = RealSkyDataPipeline(
        data_dir="data/real_sky_images",
        station_latitude=37.4275,   # Stanford University
        station_longitude=-122.1697,
        elevation_m=32.0,
        image_size=128
    )

    print("\n[1/2] Downloading and extracting real sky camera archives...")
    counts = pipeline.download_and_extract_stanford_real_images()
    for day, num in counts.items():
        print(f"  {day}: {num} real fish-eye sky frames.")

    print("\n[2/2] Generating synchronized real dataset manifest with solar physics...")
    df = pipeline.build_real_dataset_manifest(output_csv_path="data/real_dataset_manifest.csv")
    print(f"\nSuccessfully created `data/real_dataset_manifest.csv` with {len(df)} real sky samples.")
    print("=" * 80)


if __name__ == "__main__":
    main()
