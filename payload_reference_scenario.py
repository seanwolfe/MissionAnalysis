from __future__ import annotations

"""Reusable reference payload and asteroid scenario.

The numerical values in this module are copied from the current
``payload_generate_snr_contour_map(1).py`` configuration so that the contour
map and boresight optimizer can use one common payload/asteroid definition.

The representative asteroid speed is the apparent angular speed used by the
SNR streak-aperture model. The representative inertial velocity vector is also
retained for the finite Cartesian state required by ``ObservationGeometry``;
the angular-rate override takes precedence during the SNR calculation.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from payload_asteroid_snr_model import (
    AsteroidProperties,
    EnvironmentConfig,
    PayloadConfig,
    SNROptions,
)


FloatArray = NDArray[np.float64]

# =============================================================================
# Spectral response copied from the current contour-map configuration
# =============================================================================

VEGA_FLUX_DENSITY_W_M2_M = 3.68e-2
VEGA_REFERENCE_WAVELENGTH_M = 0.555e-6

QE_WAVELENGTH_M = np.array([
    358.91e-9, 369.55e-9, 379.31e-9, 389.06e-9, 399.71e-9,
    409.46e-9, 419.22e-9, 429.86e-9, 439.61e-9, 449.37e-9,
    460.01e-9, 469.76e-9, 479.52e-9, 489.27e-9, 499.92e-9,
    509.67e-9, 519.43e-9, 529.18e-9, 539.82e-9, 549.58e-9,
    559.33e-9, 569.09e-9, 579.73e-9, 589.49e-9, 600.13e-9,
    609.88e-9, 619.64e-9, 630.28e-9, 640.03e-9, 649.79e-9,
    659.54e-9, 670.19e-9, 679.94e-9, 689.70e-9, 700.34e-9,
    710.09e-9, 719.85e-9, 729.60e-9, 740.24e-9, 750.00e-9,
    759.76e-9, 769.51e-9, 780.15e-9, 789.91e-9, 799.66e-9,
    810.30e-9, 820.06e-9, 829.81e-9, 840.46e-9, 850.21e-9,
    860.85e-9, 870.61e-9, 880.36e-9, 890.12e-9, 900.76e-9,
    910.51e-9, 920.27e-9, 930.91e-9, 940.67e-9, 950.42e-9,
    960.18e-9, 969.93e-9, 980.57e-9, 990.33e-9, 1000.97e-9,
    1015.16e-9, 1030.24e-9, 1039.99e-9, 1055.07e-9,
    1064.82e-9, 1075.46e-9, 1085.22e-9,
], dtype=float)

QE_VALUES = np.array([
    0.740290, 0.736752, 0.736752, 0.926036, 0.956109, 0.959647,
    0.959647, 0.957878, 0.957878, 0.957878, 0.963185, 0.961416,
    0.957878, 0.950802, 0.945495, 0.945495, 0.949033, 0.949033,
    0.950802, 0.950802, 0.954340, 0.954340, 0.954340, 0.954340,
    0.954340, 0.949033, 0.949033, 0.941957, 0.936650, 0.927805,
    0.915422, 0.906577, 0.906577, 0.894194, 0.853507, 0.842892,
    0.828740, 0.828740, 0.775670, 0.749135, 0.727907, 0.727907,
    0.683682, 0.660685, 0.632381, 0.604077, 0.584617, 0.559851,
    0.531547, 0.508550, 0.483784, 0.457249, 0.457249, 0.427176,
    0.388258, 0.367030, 0.367030, 0.321035, 0.305114, 0.285655,
    0.285655, 0.260889, 0.218433, 0.200743, 0.181284, 0.161825,
    0.121138, 0.121138, 0.082220, 0.064529, 0.050377, 0.039763,
], dtype=float)

LUMIO_WAVELENGTH_M = 1.0e-9 * np.array(
    [400.0, 420.0, 500.0, 600.0, 700.0, 800.0, 950.0],
    dtype=float,
)

LUMIO_THROUGHPUT_KNOTS = np.array(
    [0.05, 0.35, 0.85, 0.87, 0.94, 0.89, 0.96],
    dtype=float,
)

LUMIO_THROUGHPUT = np.interp(
    QE_WAVELENGTH_M,
    LUMIO_WAVELENGTH_M,
    LUMIO_THROUGHPUT_KNOTS,
    left=0.0,
    right=0.0,
)

# =============================================================================
# Latest representative asteroid assumptions from the contour-map script
# =============================================================================

ASTEROID_ABSOLUTE_MAGNITUDE = 30.0
ASTEROID_GEOMETRIC_ALBEDO = 0.15
ASTEROID_G12 = 0.64

ASTEROID_APPARENT_SPEED_ARCSEC_PER_HOUR = 600.0
ASTEROID_APPARENT_SPEED_ARCSEC_PER_SECOND = (
    ASTEROID_APPARENT_SPEED_ARCSEC_PER_HOUR / 3600.0
)

# Retained from the contour-map input. Because the angular-rate override is
# supplied, this vector does not control the streak speed in the SNR model.
ASTEROID_VELOCITY_GEO_EME_KM_S = np.array(
    [0.15, 0.45, 0.02],
    dtype=float,
)


@dataclass(frozen=True)
class ReferencePayloadScenario:
    """Complete reusable payload, asteroid, and SNR scenario."""

    payload: PayloadConfig
    asteroid: AsteroidProperties
    environment: EnvironmentConfig
    options: SNROptions
    asteroid_velocity_geo_eme_km_s: FloatArray
    asteroid_angular_rate_override_arcsec_s: float


def make_reference_payload() -> PayloadConfig:
    """Return the current reference payload used by the contour-map script."""

    return PayloadConfig(
        exposure_time_s=65.0,
        aperture_diameter_m=0.198,
        focal_length_m=1.586,
        pixel_scale_arcsec_per_px=0.91,
        pixel_pitch_m=7.0e-6,
        psf_sigma_px=1.1,
        quantum_efficiency=QE_VALUES.copy(),
        optical_throughput=LUMIO_THROUGHPUT.copy(),
        wavelength_m=QE_WAVELENGTH_M.copy(),
        dark_current_e_per_s_px=0.069,
        read_noise_e_rms_per_px=5.0,
        background_surface_brightness_mag_arcsec2=22.0,
        wavelength_lower_m=None,
        wavelength_upper_m=None,
        spectral_samples=2001,
        zero_point_mag=None,
        vega_flux_density_w_m2_m=VEGA_FLUX_DENSITY_W_M2_M,
        vega_reference_wavelength_m=VEGA_REFERENCE_WAVELENGTH_M,
    )


def make_reference_asteroid() -> AsteroidProperties:
    """Return the current representative asteroid properties."""

    return AsteroidProperties(
        absolute_magnitude=ASTEROID_ABSOLUTE_MAGNITUDE,
        geometric_albedo=ASTEROID_GEOMETRIC_ALBEDO,
        g12=ASTEROID_G12,
    )


def make_reference_environment() -> EnvironmentConfig:
    """Return the environment configuration used by the contour map."""

    return EnvironmentConfig()


def make_reference_snr_options() -> SNROptions:
    """Return the SNR switches used by the current contour map."""

    return SNROptions(
        include_earth_double_reflection=True,
        include_moon_double_reflection=True,
        include_earth_stray_light=True,
        include_moon_stray_light=True,
        aperture_pixel_mode="continuous",
        validate_inputs=True,
    )


def make_reference_payload_scenario() -> ReferencePayloadScenario:
    """Build a fresh complete reference scenario."""

    return ReferencePayloadScenario(
        payload=make_reference_payload(),
        asteroid=make_reference_asteroid(),
        environment=make_reference_environment(),
        options=make_reference_snr_options(),
        asteroid_velocity_geo_eme_km_s=(
            ASTEROID_VELOCITY_GEO_EME_KM_S.copy()
        ),
        asteroid_angular_rate_override_arcsec_s=float(
            ASTEROID_APPARENT_SPEED_ARCSEC_PER_SECOND
        ),
    )
