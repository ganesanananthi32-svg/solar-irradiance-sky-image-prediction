"""Clear-Sky Irradiance Models (Ineichen-Perez, Haurwitz, and pvlib wrappers).

Provides theoretical clear-sky Global Horizontal Irradiance (GHI_cs),
Direct Normal Irradiance (DNI_cs), and Diffuse Horizontal Irradiance (DHI_cs),
and calculates the Clear-Sky Index (k_t) and solar irradiance residuals.
"""

import math
import numpy as np
from typing import Union, Tuple, Optional


class ClearSkyModel:
    """Base interface for Clear Sky Solar Irradiance Models."""

    def calculate_ghi(
        self,
        zenith_deg: Union[float, np.ndarray],
        day_of_year: int = 180,
        **kwargs
    ) -> Union[float, np.ndarray]:
        """Calculates theoretical clear-sky Global Horizontal Irradiance in W/m^2."""
        raise NotImplementedError

    def calculate_clear_sky_index(
        self,
        ghi_actual: Union[float, np.ndarray],
        ghi_cs: Union[float, np.ndarray],
        epsilon: float = 1e-4
    ) -> Union[float, np.ndarray]:
        """Calculates the Clear-Sky Index (k_t = GHI / GHI_cs).

        Args:
            ghi_actual: Observed pyranometer GHI (W/m^2).
            ghi_cs: Model clear-sky GHI (W/m^2).
            epsilon: Small floor to prevent division by zero near sunset/sunrise.

        Returns:
            kt: Clear-sky index (typically ~1.0 for clear sky, <0.3 for heavy clouds, >1.0 for cloud enhancement).
        """
        ghi_cs_clipped = np.maximum(ghi_cs, epsilon)
        kt = ghi_actual / ghi_cs_clipped
        return kt

    def calculate_residual(
        self,
        ghi_actual: Union[float, np.ndarray],
        ghi_cs: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Calculates the Solar Irradiance Residual: Delta GHI = GHI_actual - GHI_cs.

        Negative residual indicates cloud occlusion / attenuation.
        Positive residual indicates circumsolar cloud-edge enhancement.
        """
        return ghi_actual - ghi_cs


class HaurwitzModel(ClearSkyModel):
    """Haurwitz analytical empirical Clear Sky Model (1945/1946).

    A robust, zero-parameter model widely used as a standard baseline in solar forecasting.
    GHI_cs = 1098 * cos(zenith) * exp(-0.057 / cos(zenith)).
    """

    def calculate_ghi(
        self,
        zenith_deg: Union[float, np.ndarray],
        day_of_year: int = 180,
        **kwargs
    ) -> Union[float, np.ndarray]:
        zenith_rad = np.radians(zenith_deg)
        cos_z = np.cos(zenith_rad)

        # Clip negative cosine (below horizon)
        cos_z_clipped = np.maximum(cos_z, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            ghi_cs = 1098.0 * cos_z_clipped * np.exp(-0.057 / np.maximum(cos_z_clipped, 1e-5))

        # Zero out nighttime / below horizon
        if isinstance(zenith_deg, (int, float)):
            return float(ghi_cs if zenith_deg < 89.0 else 0.0)
        else:
            ghi_cs[zenith_deg >= 89.0] = 0.0
            return ghi_cs


class IneichenModel(ClearSkyModel):
    """Ineichen-Perez Clear Sky Model (2002).

    Incorporates station altitude and Linke Turbidity (TL) to model atmospheric
    attenuation due to water vapor, aerosols, and Rayleigh scattering.
    """

    def __init__(
        self,
        elevation_m: float = 0.0,
        default_linke_turbidity: float = 3.0
    ):
        self.elevation = elevation_m
        self.default_tl = default_linke_turbidity
        # Relative atmospheric pressure
        self.p_ratio = math.exp(-elevation_m / 8434.5)

    def calculate_ghi(
        self,
        zenith_deg: Union[float, np.ndarray],
        day_of_year: int = 180,
        linke_turbidity: Optional[float] = None,
        **kwargs
    ) -> Union[float, np.ndarray]:
        tl = linke_turbidity if linke_turbidity is not None else self.default_tl

        # Extraterrestrial normal irradiance I0 (Spencer formula)
        day_angle = 2.0 * math.pi * (day_of_year - 1) / 365.0
        eccentricity = 1.000110 + 0.034221 * math.cos(day_angle) + 0.001280 * math.sin(day_angle) + \
                       0.000719 * math.cos(2.0 * day_angle) + 0.000077 * math.sin(2.0 * day_angle)
        i0 = 1367.0 * eccentricity

        is_scalar = isinstance(zenith_deg, (int, float))
        zenith = np.array([zenith_deg], dtype=float) if is_scalar else np.array(zenith_deg, dtype=float)

        zenith_rad = np.radians(zenith)
        cos_z = np.cos(zenith_rad)

        # Kasten-Young (1989) air mass formulation
        elev_deg = 90.0 - zenith
        elev_clipped = np.maximum(elev_deg, 0.01)
        air_mass = (1.0 / (cos_z + 0.50572 * np.power(elev_clipped + 6.07995, -1.6364))) * self.p_ratio
        air_mass = np.maximum(air_mass, 1.0)

        # Ineichen altitude correction factors
        c_g1 = (5.09e-5 * self.elevation + 0.868)
        c_g2 = 0.0392 - 1.25e-5 * self.elevation

        # Ineichen GHI formulation
        fh1 = math.exp(-self.elevation / 8000.0)
        fh2 = math.exp(-self.elevation / 1250.0)

        exponent = -c_g2 * air_mass * (fh1 + fh2 * (tl - 1.0))
        ghi = c_g1 * i0 * cos_z * np.exp(exponent) * np.exp(0.01 * np.power(air_mass, 1.8))

        # Mask below horizon
        ghi[zenith >= 89.0] = 0.0
        ghi = np.maximum(ghi, 0.0)

        return float(ghi[0]) if is_scalar else ghi
