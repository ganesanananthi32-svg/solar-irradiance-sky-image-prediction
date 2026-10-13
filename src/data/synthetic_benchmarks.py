"""Physics-Grounded Synthetic Solar Benchmark Generator.

Generates physically consistent all-sky images and synchronized pyranometer
GHI/DNI measurements with:
- Exact solar position trajectories across the sky dome.
- Theoretical Ineichen clear-sky irradiance.
- Dynamic synthetic cloud fields with wind advection (clear, broken cumulus, overcast).
- Circumsolar direct beam attenuation and cloud lensing enhancements.

Enables immediate zero-setup end-to-end benchmarking, training, and testing
without waiting for 50GB external dataset downloads.
"""

import numpy as np
import datetime
from typing import Dict, List, Tuple
from ..physics.solar_geometry import SolarGeometry
from ..physics.clear_sky_models import IneichenModel
from ..vision.image_preprocessing import SkyImagePreprocessor
from ..vision.circumsolar_extractor import CircumsolarExtractor


class SyntheticSolarBenchmarkGenerator:
    """Generates synthetic paired sky images, clear-sky baselines, and ground GHI."""

    def __init__(
        self,
        station_latitude: float = 37.4275,  # Stanford
        station_longitude: float = -122.1697,
        elevation_m: float = 32.0,
        image_size: int = 128,
        seed: int = 42
    ):
        np.random.seed(seed)
        self.geom = SolarGeometry(station_latitude, station_longitude, elevation_m)
        self.csm = IneichenModel(elevation_m=elevation_m, default_linke_turbidity=3.0)
        self.preprocessor = SkyImagePreprocessor(target_size=(image_size, image_size))
        self.circ_extractor = CircumsolarExtractor(self.geom, roi_size=(48, 48))
        self.img_size = image_size

    def generate_day_sequence(
        self,
        day_date: datetime.date = datetime.date(2024, 6, 21),
        start_hour: int = 8,
        end_hour: int = 17,
        step_minutes: int = 5,
        weather_type: str = "broken_clouds"
    ) -> Dict[str, np.ndarray]:
        """Generates a full daytime sequence of sky images and irradiance readings.

        Args:
            day_date: Calendar date to simulate.
            start_hour: Morning hour (solar day).
            end_hour: Evening hour.
            step_minutes: Temporal resolution (e.g., 1 min, 5 min).
            weather_type: 'clear', 'broken_clouds', 'overcast', 'ramp_storm'.

        Returns:
            Dictionary containing:
                - images: (N, H, W, 3) normalized float32 sky images
                - circumsolar_crops: (N, 48, 48, 3) sun-centered patches
                - ghi_actual: (N,) ground-truth pyranometer GHI
                - ghi_cs: (N,) theoretical Ineichen clear-sky GHI
                - aux_features: (N, 5) [zenith, azimuth, cos_zenith, air_temp, humidity]
                - timestamps: List of ISO timestamps
        """
        timestamps = []
        cur_dt = datetime.datetime.combine(day_date, datetime.time(start_hour, 0), tzinfo=datetime.timezone.utc)
        end_dt = datetime.datetime.combine(day_date, datetime.time(end_hour, 0), tzinfo=datetime.timezone.utc)

        dt_list = []
        while cur_dt <= end_dt:
            dt_list.append(cur_dt)
            cur_dt += datetime.timedelta(minutes=step_minutes)

        n = len(dt_list)
        images = np.zeros((n, self.img_size, self.img_size, 3), dtype=np.float32)
        circ_crops = np.zeros((n, 48, 48, 3), dtype=np.float32)
        ghi_actual = np.zeros(n, dtype=np.float32)
        ghi_cs = np.zeros(n, dtype=np.float32)
        aux_features = np.zeros((n, 5), dtype=np.float32)

        # Sky dome mask
        sky_mask = self.preprocessor.create_circular_mask(self.img_size, self.img_size)

        # Cloud field state (2D grid advected by wind)
        cloud_grid_size = 256
        y_grid, x_grid = np.ogrid[:cloud_grid_size, :cloud_grid_size]

        # Generate procedural cloud texture using overlapping sine frequencies + noise
        cloud_texture = (
            np.sin(x_grid / 24.0) * np.cos(y_grid / 24.0)
            + 0.5 * np.sin(x_grid / 12.0 + y_grid / 12.0)
            + 0.25 * np.random.randn(cloud_grid_size, cloud_grid_size)
        )
        cloud_texture = (cloud_texture - cloud_texture.min()) / (cloud_texture.max() - cloud_texture.min())

        # Wind advection velocity (pixels per step)
        wind_vx = 3
        wind_vy = 1

        day_of_year = day_date.timetuple().tm_yday

        for i, dt in enumerate(dt_list):
            timestamps.append(dt.isoformat())
            zenith_deg, elev_deg, az_deg = self.geom.calculate_solar_position(dt)
            cs_val = self.csm.calculate_ghi(zenith_deg, day_of_year=day_of_year)
            ghi_cs[i] = cs_val

            cos_z = max(0.0, np.cos(np.radians(zenith_deg)))
            temp = 20.0 + 10.0 * np.sin(np.pi * (i / max(1, n - 1)))
            humidity = 50.0 - 15.0 * np.sin(np.pi * (i / max(1, n - 1)))
            aux_features[i] = [zenith_deg / 90.0, az_deg / 360.0, cos_z, temp / 40.0, humidity / 100.0]

            # Shift cloud field by wind advection
            shift_x = (i * wind_vx) % (cloud_grid_size - self.img_size)
            shift_y = (i * wind_vy) % (cloud_grid_size - self.img_size)
            cloud_patch = cloud_texture[shift_y : shift_y + self.img_size, shift_x : shift_x + self.img_size]

            # Adjust cloud threshold based on weather scenario
            if weather_type == "clear":
                cloud_binary = np.zeros_like(cloud_patch, dtype=bool)
                transmittance = 1.0
            elif weather_type == "overcast":
                cloud_binary = np.ones_like(cloud_patch, dtype=bool)
                transmittance = 0.22
            elif weather_type == "ramp_storm":
                # Rapid transitions between clear and thick clouds
                threshold = 0.50 + 0.35 * np.sin(i * 0.45)
                cloud_binary = cloud_patch > threshold
            else:  # broken_clouds
                cloud_binary = cloud_patch > 0.55

            # Render synthetic RGB sky image
            # Clear blue sky: R=0.35, G=0.60, B=0.92
            # Clouds: R=0.88, G=0.90, B=0.92
            img_r = np.where(cloud_binary, 0.88, 0.35)
            img_g = np.where(cloud_binary, 0.90, 0.60)
            img_b = np.where(cloud_binary, 0.92, 0.92)

            # Project sun disk onto image
            x_sun, y_sun, is_vis = self.geom.project_sun_to_fisheye(
                zenith_deg, az_deg, (self.img_size, self.img_size)
            )

            sun_blocked = False
            if is_vis and 0 <= x_sun < self.img_size and 0 <= y_sun < self.img_size:
                sy, sx = int(y_sun), int(x_sun)
                # Check if cloud occludes the sun pixel
                sun_blocked = bool(cloud_binary[sy, sx])

                # Draw intense white solar glare disk
                y_idx, x_idx = np.ogrid[:self.img_size, :self.img_size]
                dist_sun = np.sqrt((x_idx - x_sun) ** 2 + (y_idx - y_sun) ** 2)

                if not sun_blocked:
                    # Unblocked bright solar glare
                    glare = np.exp(-dist_sun / 8.0)
                    img_r = np.clip(img_r + glare * 0.9, 0, 1)
                    img_g = np.clip(img_g + glare * 0.9, 0, 1)
                    img_b = np.clip(img_b + glare * 0.9, 0, 1)
                else:
                    # Diffused circumsolar forward scattering behind cloud
                    glare = np.exp(-dist_sun / 14.0) * 0.35
                    img_r = np.clip(img_r + glare, 0, 1)
                    img_g = np.clip(img_g + glare, 0, 1)
                    img_b = np.clip(img_b + glare, 0, 1)

            # Mask perimeter outside circular lens
            img_rgb = np.stack([img_r, img_g, img_b], axis=-1).astype(np.float32)
            img_rgb[~sky_mask] = 0.0
            images[i] = img_rgb

            # Extract circumsolar crop
            crop, _ = self.circ_extractor.extract_circumsolar_roi(img_rgb, zenith_deg, az_deg)
            circ_crops[i] = crop

            # Calculate actual ground GHI based on physical direct beam attenuation + diffuse
            if weather_type == "clear":
                ghi_actual[i] = cs_val
            elif weather_type == "overcast":
                ghi_actual[i] = cs_val * transmittance
            else:
                if sun_blocked:
                    # Direct beam blocked: only diffuse sky remains (~20% to 35% of clear sky)
                    cloud_density = float(cloud_patch[int(y_sun), int(x_sun)]) if is_vis else 0.8
                    attenuation = 0.18 + (1.0 - cloud_density) * 0.15
                    ghi_actual[i] = cs_val * attenuation
                else:
                    # Sun unblocked: check if cloud edge is close -> cloud enhancement lensing effect!
                    min_dist_to_cloud = 100.0
                    if np.any(cloud_binary):
                        y_c, x_c = np.where(cloud_binary)
                        dists = np.sqrt((x_c - x_sun) ** 2 + (y_c - y_sun) ** 2)
                        min_dist_to_cloud = float(np.min(dists))

                    if 5.0 < min_dist_to_cloud < 25.0:
                        # Cloud edge circumsolar enhancement: GHI exceeds clear-sky by up to 15%!
                        enhancement = 1.0 + 0.15 * (1.0 - (min_dist_to_cloud / 25.0))
                        ghi_actual[i] = cs_val * enhancement
                    else:
                        ghi_actual[i] = cs_val

        return {
            "images": images,
            "circumsolar_crops": circ_crops,
            "ghi_actual": ghi_actual,
            "ghi_cs": ghi_cs,
            "aux_features": aux_features,
            "timestamps": np.array(timestamps)
        }
