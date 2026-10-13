"""Step 2: Visualize Celestial Sun Tracking, Circumsolar ROI, and Polar Warping.

Demonstrates the 2026+ Sun-Centric Coordinate Transformation on real sky imagery.
Saves visual diagnostics to `results/real_data/real_sun_polar_warp_sample.png`.
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.physics.solar_geometry import SolarGeometry
from src.vision.image_preprocessing import SkyImagePreprocessor
from src.vision.circumsolar_extractor import CircumsolarExtractor
from src.models.diff_csm_2026 import SunCentricPolarWarper


def main():
    print("=" * 80)
    print("STEP 2: REAL-IMAGE SUN-TRACKING & SUN-CENTRIC POLAR WARPING DIAGNOSTICS")
    print("=" * 80)

    csv_path = "data/real_dataset_manifest.csv"
    if not os.path.exists(csv_path):
        print("Dataset manifest not found. Please run Step 1 first!")
        return

    df = pd.read_csv(csv_path)
    # Pick a sample near midday where sun is visible
    sample_row = df.iloc[len(df) // 3]
    img_path = sample_row["image_path"]
    zenith_deg = sample_row["zenith_deg"]
    az_deg = sample_row["azimuth_deg"]

    print(f"Loading real sky image: {img_path}")
    print(f"Solar Position: Zenith={zenith_deg:.1f} deg, Azimuth={az_deg:.1f} deg")

    geom = SolarGeometry(37.4275, -122.1697, 32.0)
    prep = SkyImagePreprocessor(target_size=(128, 128))
    circ = CircumsolarExtractor(geom, roi_size=(48, 48))
    warper = SunCentricPolarWarper(polar_radial_bins=48, polar_angular_bins=64)

    raw_img = Image.open(img_path).convert("RGB")
    norm_img, mask = prep.preprocess_image(raw_img)
    x_sun, y_sun, is_vis = geom.project_sun_to_fisheye(zenith_deg, az_deg, (128, 128))
    crop, metrics = circ.extract_circumsolar_roi(norm_img, zenith_deg, az_deg)
    polar_map = warper.warp_image_to_sun_polar(norm_img, x_sun, y_sun, max_radius=50.0)

    # Render Multi-Panel Visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)

    # Panel 1: Original Real Sky Image + Detected Sun Marker
    axes[0].imshow(norm_img)
    axes[0].scatter([x_sun], [y_sun], color="red", s=80, marker="o", edgecolors="yellow", linewidths=1.5, label="Astronomical Sun")
    # Draw circumsolar bounding box
    r_box = 24
    rect = plt.Rectangle((x_sun - r_box, y_sun - r_box), 2*r_box, 2*r_box, fill=False, color="cyan", linestyle="--", linewidth=1.5, label="Circumsolar +-15 deg")
    axes[0].add_patch(rect)
    axes[0].set_title("1. Real Fisheye Sky Image\n(Stanford Rooftop Camera)", fontsize=10, fontweight="bold")
    axes[0].axis("off")
    axes[0].legend(loc="lower left", fontsize=8)

    # Panel 2: Red-to-Blue Ratio (RBR) Cloud Segmentation
    r = norm_img[..., 0]
    b = norm_img[..., 2]
    rbr = r / (b + 1e-4)
    rbr[~mask] = 0.0
    im2 = axes[1].imshow(rbr, cmap="coolwarm")
    axes[1].set_title("2. Red-Blue Ratio (RBR)\nCloud Mie Scattering Map", fontsize=10, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Sun-Centered Circumsolar Cone Crop
    axes[2].imshow(crop)
    axes[2].scatter([24], [24], color="yellow", s=50, marker="*")
    axes[2].set_title(f"3. Circumsolar Cone ROI (48x48)\nOcclusion={metrics['sun_occlusion_score']:.2f}", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: 2026+ Sun-Centric Log-Polar Warping Map
    axes[3].imshow(polar_map)
    axes[3].set_title("4. Sun-Centric Polar Map (r, theta)\n(Clouds Move Down r -> 0)", fontsize=10, fontweight="bold")
    axes[3].set_xlabel("Azimuth Angle around Sun (theta)")
    axes[3].set_ylabel("Radial Distance from Sun (r)")

    os.makedirs("results/real_data", exist_ok=True)
    out_path = "results/real_data/real_sun_polar_warp_sample.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved real diagnostic visualization to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
