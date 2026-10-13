"""Solar Geometry Calculator and Fisheye Camera Sun-Tracking Projection.

Provides astronomical equations for solar position (zenith, azimuth, elevation)
and maps celestial sun coordinates to 2D fisheye sky camera image pixels.
"""

import math
import numpy as np
from datetime import datetime, timezone
from typing import Tuple, Union, Optional


class SolarGeometry:
    """Astronomical solar position and fisheye camera projection utility."""

    def __init__(
        self,
        latitude: float,
        longitude: float,
        elevation_m: float = 0.0,
        camera_north_offset_deg: float = 0.0,
        fisheye_lens_type: str = "equidistant"
    ):
        """
        Args:
            latitude: Station latitude in degrees (-90 to +90).
            longitude: Station longitude in degrees (-180 to +180).
            elevation_m: Station elevation in meters above sea level.
            camera_north_offset_deg: Azimuth alignment offset of the camera (0 = North up).
            fisheye_lens_type: Projection type ('equidistant', 'equisolid', 'orthographic').
        """
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation_m
        self.camera_north_offset = camera_north_offset_deg
        self.lens_type = fisheye_lens_type

    def calculate_solar_position(
        self,
        dt: datetime
    ) -> Tuple[float, float, float]:
        """Calculates solar zenith angle, elevation angle, and azimuth angle.

        Args:
            dt: UTC datetime or timezone-aware datetime.

        Returns:
            zenith_deg: Solar zenith angle in degrees (0 = overhead).
            elevation_deg: Solar elevation angle in degrees (90 - zenith).
            azimuth_deg: Solar azimuth angle in degrees (0 = North, 90 = East, 180 = South, 270 = West).
        """
        # Ensure UTC
        if dt.tzinfo is not None:
            dt_utc = dt.astimezone(timezone.utc)
        else:
            dt_utc = dt

        # Day of year (1 to 365/366)
        day_of_year = dt_utc.timetuple().tm_yday
        hour_float = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0

        # Fractional year in radians
        gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hour_float - 12.0) / 24.0)

        # Equation of Time (EoT) in minutes (Spencer 1971 formulation)
        eot = 229.18 * (
            0.000075
            + 0.001868 * math.cos(gamma)
            - 0.032077 * math.sin(gamma)
            - 0.014615 * math.cos(2.0 * gamma)
            - 0.040849 * math.sin(2.0 * gamma)
        )

        # Solar Declination in radians
        delta = (
            0.006918
            - 0.399912 * math.cos(gamma)
            + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2.0 * gamma)
            + 0.000907 * math.sin(2.0 * gamma)
            - 0.002697 * math.cos(3.0 * gamma)
            + 0.001480 * math.sin(3.0 * gamma)
        )

        # True Solar Time (TST) in minutes
        time_offset = eot + 4.0 * self.longitude
        tst = hour_float * 60.0 + time_offset

        # Solar Hour Angle (omega) in degrees
        omega = (tst / 4.0) - 180.0
        omega_rad = math.radians(omega)
        phi_rad = math.radians(self.latitude)

        # Cosine of Solar Zenith Angle
        cos_zenith = math.sin(phi_rad) * math.sin(delta) + math.cos(phi_rad) * math.cos(delta) * math.cos(omega_rad)
        cos_zenith = max(-1.0, min(1.0, cos_zenith))
        zenith_rad = math.acos(cos_zenith)
        zenith_deg = math.degrees(zenith_rad)
        elevation_deg = 90.0 - zenith_deg

        # Solar Azimuth Angle
        if zenith_deg >= 90.0:
            sin_elev = math.sin(math.radians(max(0.001, elevation_deg)))
        else:
            sin_elev = math.sin(math.radians(elevation_deg))

        # Azimuth calculation (clockwise from North)
        cos_az = (math.sin(delta) * math.cos(phi_rad) - math.cos(delta) * math.sin(phi_rad) * math.cos(omega_rad)) / math.sin(zenith_rad) if zenith_rad > 1e-4 else 1.0
        cos_az = max(-1.0, min(1.0, cos_az))
        azimuth_deg = math.degrees(math.acos(cos_az))

        if omega > 0:
            azimuth_deg = (360.0 - azimuth_deg) % 360.0
        else:
            azimuth_deg = azimuth_deg % 360.0

        return zenith_deg, elevation_deg, azimuth_deg

    def project_sun_to_fisheye(
        self,
        zenith_deg: float,
        azimuth_deg: float,
        image_shape: Tuple[int, int],
        mask_radius: Optional[float] = None
    ) -> Tuple[float, float, bool]:
        """Projects celestial sun position (zenith, azimuth) to 2D image pixel coordinates (x, y).

        Standard equidistant lens model: r = R_mask * (zenith / 90.0).

        Args:
            zenith_deg: Solar zenith angle in degrees.
            azimuth_deg: Solar azimuth angle in degrees.
            image_shape: (height, width) of the sky image.
            mask_radius: Optional pixel radius of the circular sky dome mask.

        Returns:
            (x, y, is_visible):
                x: horizontal pixel coordinate (column).
                y: vertical pixel coordinate (row).
                is_visible: True if sun is above the local horizon (zenith <= 90 deg).
        """
        h, w = image_shape
        center_x = w / 2.0
        center_y = h / 2.0
        if mask_radius is None:
            mask_radius = min(h, w) / 2.0 * 0.95

        if zenith_deg > 90.0:
            # Below horizon (nighttime or dawn/dusk)
            return center_x, center_y, False

        # Lens projection mapping
        if self.lens_type == "equidistant":
            # r proportional to zenith
            r = mask_radius * (zenith_deg / 90.0)
        elif self.lens_type == "equisolid":
            # r = 2 * f * sin(zenith / 2)
            r = mask_radius * (math.sin(math.radians(zenith_deg / 2.0)) / math.sin(math.radians(45.0)))
        else:
            r = mask_radius * (zenith_deg / 90.0)

        # Azimuth angle relative to camera orientation
        phi = math.radians(azimuth_deg - self.camera_north_offset)

        # Standard navigation coords: 0 deg = North (up, negative Y), 90 deg = East (right, positive X)
        x = center_x + r * math.sin(phi)
        y = center_y - r * math.cos(phi)

        return float(x), float(y), True
