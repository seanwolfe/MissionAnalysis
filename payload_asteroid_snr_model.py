
from __future__ import annotations

"""
Asteroid signal-to-noise ratio model.

This module implements the electron-count SNR model described by

    SNR = N_A / sqrt(
        N_A
        + N_bg
        + N_dark
        + sigma_rn,A^2
        + N_sl
    )

where the asteroid signal includes direct reflected sunlight and optional
secondary illumination from Earth and the Moon. The noise model includes
source shot noise, sky background, dark current, read noise, and optional
Earth/Moon stray light.

Key conventions
---------------
1. Cartesian position inputs use kilometres and velocity inputs use km/s.
   Every state in one evaluation must use the same origin, axes, frame, and
   epoch. Vector inputs have shape (..., 3), while scalar quantities have
   shape (...) and are broadcast over the common batch shape.

2. Public angular inputs are in radians, except for quantities whose names
   explicitly include ``arcsec``.

3. The apparent angular speed is calculated from the observer and asteroid
   states and is assumed constant over each exposure. The payload is assumed
   approximately inertially fixed during an exposure.

4. ``zero_point_mag`` is the apparent magnitude of a source that produces
   exactly 1 electron per second in the complete observation system. Thus,

       electron_rate_e_s = 10**[-0.4 * (magnitude - zero_point_mag)]

   If a zero point is supplied, it takes precedence over Vega-based
   calculation.

5. Spectral quantities may be scalars or one-dimensional arrays:
   - A scalar QE, optical throughput, or Vega flux is assumed constant over
     the optical band.
   - An array is interpreted as samples of a wavelength-dependent function
     on ``wavelength_m``.
   - Scalar and array spectral inputs may be mixed.
   - All wavelength integrals use composite trapezoidal integration.

6. The adopted stray-light efficiency is wavelength independent and is
   interpolated linearly in log10(zeta) through:

       (0 deg, 1.0)
       (5 deg, 2.45e-7)
       (10 deg, 5.16e-8)
       (15 deg, 2.45e-8)

   The value is held constant at 2.45e-8 for off-axis angles >= 15 deg.
   This is a residual-ghost-based approximation; separate roughness and
   contamination terms are not modelled.

7. The HG12 basis functions are cubic splines through the supplied tabulated
   values. Small negative spline artefacts are clipped to zero.

Dependencies
------------
numpy
scipy
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import Boltzmann as BOLTZMANN_CONSTANT_J_K
from scipy.constants import c as SPEED_OF_LIGHT_M_S
from scipy.constants import h as PLANCK_CONSTANT_J_S
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline


FloatArray: TypeAlias = NDArray[np.float64]

AU_KM = 149_597_870.7
ARCSEC_PER_RADIAN = 206_264.80624709636

# Visible, overridable preliminary defaults.
DEFAULT_SUN_RADIUS_KM = 695_700.0
DEFAULT_SUN_TEMPERATURE_K = 5_772.0
DEFAULT_EARTH_RADIUS_KM = 6_371.0
DEFAULT_EARTH_GEOMETRIC_ALBEDO = 0.434
DEFAULT_MOON_RADIUS_KM = 1_737.4
DEFAULT_MOON_GEOMETRIC_ALBEDO = 0.12


@dataclass(frozen=True)
class PayloadConfig:
    """Optical payload and detector configuration.

    Spectral inputs
    ---------------
    ``quantum_efficiency`` and ``optical_throughput`` may each be either:
    - a scalar, interpreted as constant over the complete wavelength band; or
    - a one-dimensional array sampled on ``wavelength_m``.

    If either response is an array, ``wavelength_m`` must be supplied and
    every spectral array must have the same length. If both are scalars,
    ``wavelength_m`` may be omitted and an internal uniform grid is generated
    from ``wavelength_lower_m`` and ``wavelength_upper_m``.

    Vega zero point
    ---------------
    If ``zero_point_mag`` is not None, it is used directly and all Vega fields
    are ignored.

    If ``vega_flux_density_w_m2_m`` is a scalar, it is treated as a
    monochromatic energy-flux density evaluated at
    ``vega_reference_wavelength_m``. The corresponding photon-flux density is
    assumed constant over the band, following the approximation in the
    derivation.

    If ``vega_flux_density_w_m2_m`` is an array, it is treated as a sampled
    Vega energy-flux-density spectrum on ``wavelength_m`` and the full
    wavelength-dependent integral is evaluated.
    """

    exposure_time_s: float
    aperture_diameter_m: float
    focal_length_m: float
    pixel_scale_arcsec_per_px: float
    pixel_pitch_m: float
    psf_sigma_px: ArrayLike

    quantum_efficiency: float | ArrayLike
    optical_throughput: float | ArrayLike

    dark_current_e_per_s_px: float
    read_noise_e_rms_per_px: float
    background_surface_brightness_mag_arcsec2: float

    wavelength_m: ArrayLike | None = None
    wavelength_lower_m: float | None = None
    wavelength_upper_m: float | None = None
    spectral_samples: int = 2001

    zero_point_mag: float | None = None
    vega_flux_density_w_m2_m: float | ArrayLike | None = None
    vega_reference_wavelength_m: float | None = None


@dataclass(frozen=True)
class AsteroidProperties:
    """Intrinsic asteroid properties.

    Each field may be a scalar or a batch array. The fields must be
    broadcastable to the observation batch shape.
    """

    absolute_magnitude: ArrayLike
    geometric_albedo: ArrayLike
    g12: ArrayLike


@dataclass(frozen=True)
class ObservationGeometry:
    """Observer, asteroid, Sun, Earth, and Moon geometry.

    Vector fields must have shape (..., 3). Scalar/batch fields must have
    shape (...). All inputs are broadcast over a common batch shape.

    ``asteroid_angular_rate_arcsec_s`` is optional. When supplied, it
    overrides the state-derived inertial line-of-sight angular speed.
    """

    observer_position_km: ArrayLike
    observer_velocity_km_s: ArrayLike

    asteroid_position_km: ArrayLike
    asteroid_velocity_km_s: ArrayLike

    sun_position_km: ArrayLike
    earth_position_km: ArrayLike
    moon_position_km: ArrayLike

    boresight_unit_vector: ArrayLike

    asteroid_angular_rate_arcsec_s: ArrayLike | None = None


@dataclass(frozen=True)
class BodyProperties:
    """Photometric properties of an illuminating/disturbing body."""

    radius_km: float
    geometric_albedo: float
    phase_law: Literal["lambert", "lommel_seeliger"]


@dataclass(frozen=True)
class EnvironmentConfig:
    """Solar, Earth, and Moon constants used by the model."""

    solar_radius_km: float = DEFAULT_SUN_RADIUS_KM
    solar_temperature_k: float = DEFAULT_SUN_TEMPERATURE_K
    earth: BodyProperties = BodyProperties(
        radius_km=DEFAULT_EARTH_RADIUS_KM,
        geometric_albedo=DEFAULT_EARTH_GEOMETRIC_ALBEDO,
        phase_law="lambert",
    )
    moon: BodyProperties = BodyProperties(
        radius_km=DEFAULT_MOON_RADIUS_KM,
        geometric_albedo=DEFAULT_MOON_GEOMETRIC_ALBEDO,
        phase_law="lommel_seeliger",
    )


@dataclass(frozen=True)
class SNROptions:
    """Optional model switches."""

    include_earth_double_reflection: bool = True
    include_moon_double_reflection: bool = True
    include_earth_stray_light: bool = True
    include_moon_stray_light: bool = True

    aperture_pixel_mode: Literal["continuous", "ceil"] = "continuous"
    validate_inputs: bool = True


@dataclass(frozen=True)
class HG12PhaseTerms:
    g1: FloatArray
    g2: FloatArray
    phi1: FloatArray
    phi2: FloatArray
    phi3: FloatArray
    phase_function: FloatArray


@dataclass(frozen=True)
class ApertureTerms:
    trail_length_px: FloatArray
    aperture_length_px: FloatArray
    aperture_width_px: FloatArray
    n_pixels: FloatArray


@dataclass(frozen=True)
class GeometryTerms:
    asteroid_sun_distance_km: FloatArray
    asteroid_observer_distance_km: FloatArray
    asteroid_earth_distance_km: FloatArray
    asteroid_moon_distance_km: FloatArray
    earth_sun_distance_km: FloatArray
    moon_sun_distance_km: FloatArray
    observer_earth_distance_km: FloatArray
    observer_moon_distance_km: FloatArray

    direct_phase_angle_rad: FloatArray
    earth_asteroid_phase_angle_rad: FloatArray
    moon_asteroid_phase_angle_rad: FloatArray
    sun_earth_asteroid_phase_angle_rad: FloatArray
    sun_moon_asteroid_phase_angle_rad: FloatArray
    sun_earth_observer_phase_angle_rad: FloatArray
    sun_moon_observer_phase_angle_rad: FloatArray

    asteroid_off_axis_angle_rad: FloatArray
    earth_off_axis_angle_rad: FloatArray
    moon_off_axis_angle_rad: FloatArray

    apparent_angular_speed_arcsec_s: FloatArray


@dataclass(frozen=True)
class SNRResult:
    """Complete SNR result and intermediate diagnostic quantities."""

    snr: FloatArray

    absolute_magnitude: FloatArray
    apparent_magnitude: FloatArray
    geometric_albedo: FloatArray
    asteroid_diameter_m: FloatArray
    asteroid_radius_km: FloatArray

    g12: FloatArray
    g1: FloatArray
    g2: FloatArray

    direct_phase_function: FloatArray
    earth_asteroid_phase_function: FloatArray
    moon_asteroid_phase_function: FloatArray

    zero_point_mag: float
    wavelength_grid_m: FloatArray
    solar_response_integral: float

    asteroid_direct_rate_e_s: FloatArray
    earth_double_reflection_rate_e_s: FloatArray
    moon_double_reflection_rate_e_s: FloatArray

    direct_signal_e: FloatArray
    earth_double_reflection_e: FloatArray
    moon_double_reflection_e: FloatArray
    signal_electrons: FloatArray

    background_electrons: FloatArray
    dark_electrons: FloatArray
    read_noise_variance_e2: FloatArray

    earth_stray_light_rate_e_s_px: FloatArray
    moon_stray_light_rate_e_s_px: FloatArray
    earth_stray_light_electrons: FloatArray
    moon_stray_light_electrons: FloatArray
    stray_light_electrons: FloatArray

    source_shot_noise_variance_e2: FloatArray
    total_noise_variance_e2: FloatArray
    total_noise_rms_e: FloatArray

    trail_length_px: FloatArray
    aperture_length_px: FloatArray
    aperture_width_px: FloatArray
    n_pixels: FloatArray
    apparent_angular_speed_arcsec_s: FloatArray

    asteroid_sun_distance_au: FloatArray
    asteroid_observer_distance_au: FloatArray

    direct_phase_angle_rad: FloatArray
    earth_asteroid_phase_angle_rad: FloatArray
    moon_asteroid_phase_angle_rad: FloatArray
    sun_earth_asteroid_phase_angle_rad: FloatArray
    sun_moon_asteroid_phase_angle_rad: FloatArray
    sun_earth_observer_phase_angle_rad: FloatArray
    sun_moon_observer_phase_angle_rad: FloatArray

    asteroid_off_axis_angle_rad: FloatArray
    earth_off_axis_angle_rad: FloatArray
    moon_off_axis_angle_rad: FloatArray


@dataclass(frozen=True)
class HG12PhaseModel:
    """Cubic-spline implementation of the supplied HG12 basis functions."""

    phi1_spline: CubicSpline
    phi2_spline: CubicSpline
    phi3_spline: CubicSpline

    @classmethod
    def from_default_table(cls) -> "HG12PhaseModel":
        """Construct the model from the supplied phase-function table.

        The spline independent variable is degrees. The supplied endpoint
        derivatives are interpreted as derivatives per radian, matching the
        older implementation, and are converted to derivatives per degree
        before being passed to SciPy.
        """

        alphas_phi_12_deg = np.array(
            [0.0, 7.5, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0],
            dtype=float,
        )
        phi1_values = np.array(
            [
                1.0,
                7.5e-1,
                3.3486016e-1,
                1.3410560e-1,
                5.1104756e-2,
                2.1465687e-2,
                3.6396989e-3,
                0.0,
            ],
            dtype=float,
        )
        phi2_values = np.array(
            [
                1.0,
                9.25e-1,
                6.2884169e-1,
                3.1755495e-1,
                1.2716367e-1,
                2.2373903e-2,
                1.6505689e-4,
                0.0,
            ],
            dtype=float,
        )

        alphas_phi_3_deg = np.array(
            [0.0, 0.3, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 30.0, 60.0, 90.0, 180.0],
            dtype=float,
        )
        phi3_values = np.array(
            [
                1.0,
                8.3381185e-1,
                5.7735424e-1,
                4.2144772e-1,
                2.3174230e-1,
                1.0348178e-1,
                6.1733473e-2,
                1.6107006e-2,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )

        phi1_derivative_initial_per_rad = -1.909859317102744
        phi1_derivative_end_per_rad = -9.1328612e-2
        phi2_derivative_initial_per_rad = -0.5729577951308232
        phi2_derivative_end_per_rad = -8.6573138e-8
        phi3_derivative_initial_per_rad = -1.0630097e-1
        phi3_derivative_end_per_rad = 0.0

        per_rad_to_per_deg = np.pi / 180.0

        phi1_spline = CubicSpline(
            alphas_phi_12_deg,
            phi1_values,
            bc_type=(
                (1, phi1_derivative_initial_per_rad * per_rad_to_per_deg),
                (1, phi1_derivative_end_per_rad * per_rad_to_per_deg),
            ),
            extrapolate=False,
        )
        phi2_spline = CubicSpline(
            alphas_phi_12_deg,
            phi2_values,
            bc_type=(
                (1, phi2_derivative_initial_per_rad * per_rad_to_per_deg),
                (1, phi2_derivative_end_per_rad * per_rad_to_per_deg),
            ),
            extrapolate=False,
        )
        phi3_spline = CubicSpline(
            alphas_phi_3_deg,
            phi3_values,
            bc_type=(
                (1, phi3_derivative_initial_per_rad * per_rad_to_per_deg),
                (1, phi3_derivative_end_per_rad * per_rad_to_per_deg),
            ),
            extrapolate=False,
        )

        return cls(
            phi1_spline=phi1_spline,
            phi2_spline=phi2_spline,
            phi3_spline=phi3_spline,
        )

    def evaluate(
        self,
        phase_angle_rad: ArrayLike,
        g12: ArrayLike,
    ) -> HG12PhaseTerms:
        """Evaluate G1, G2, the three basis functions, and their combination."""

        alpha_rad = np.asarray(phase_angle_rad, dtype=float)
        g12_array = np.asarray(g12, dtype=float)

        alpha_deg = np.rad2deg(alpha_rad)

        phi1 = np.maximum(np.asarray(self.phi1_spline(alpha_deg), dtype=float), 0.0)
        phi2 = np.maximum(np.asarray(self.phi2_spline(alpha_deg), dtype=float), 0.0)
        phi3 = np.maximum(np.asarray(self.phi3_spline(alpha_deg), dtype=float), 0.0)

        g1, g2 = g12_to_g1_g2(g12_array)

        phase_function = (
            g1 * phi1
            + g2 * phi2
            + (1.0 - g1 - g2) * phi3
        )

        # Protect against tiny negative values caused by floating-point
        # combination of spline outputs and weights. Enforce the exact
        # physical endpoint at 180 degrees.
        phase_function = np.maximum(phase_function, 0.0)
        phase_function = np.where(
            alpha_rad >= np.pi - 1.0e-12,
            0.0,
            phase_function,
        )

        return HG12PhaseTerms(
            g1=np.asarray(g1, dtype=float),
            g2=np.asarray(g2, dtype=float),
            phi1=phi1,
            phi2=phi2,
            phi3=phi3,
            phase_function=np.asarray(phase_function, dtype=float),
        )


def g12_to_g1_g2(g12: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Convert the HG12 parameter G12 to G1 and G2.

    The piecewise relation matches the earlier SnrGen implementation.
    """

    g12_array = np.asarray(g12, dtype=float)

    g1 = np.where(
        g12_array >= 0.2,
        0.9529 * g12_array + 0.02162,
        0.7527 * g12_array + 0.06164,
    )
    g2 = np.where(
        g12_array >= 0.2,
        -0.6125 * g12_array + 0.5572,
        -0.9612 * g12_array + 0.6270,
    )

    return np.asarray(g1, dtype=float), np.asarray(g2, dtype=float)


def diameter_from_absolute_magnitude(
    absolute_magnitude: ArrayLike,
    geometric_albedo: ArrayLike,
) -> FloatArray:
    """Calculate asteroid diameter in metres from H and geometric albedo.

    This follows the active equation in the derivation exactly:

        H = 15.618 - 5 log10(0.001 D_m) - 2.5 log10(a)

    rather than the slightly different coefficient in the commented
    rearrangement.
    """

    h = np.asarray(absolute_magnitude, dtype=float)
    albedo = np.asarray(geometric_albedo, dtype=float)

    diameter_m = 1000.0 * 10.0 ** (
        (
            15.618
            - h
            - 2.5 * np.log10(albedo)
        )
        / 5.0
    )
    return np.asarray(diameter_m, dtype=float)


def lambert_phase_function(phase_angle_rad: ArrayLike) -> FloatArray:
    """Lambert phase law, equal to one at zero phase and zero at pi."""

    alpha = np.asarray(phase_angle_rad, dtype=float)
    value = (
        (np.pi - alpha) * np.cos(alpha)
        + np.sin(alpha)
    ) / np.pi
    return np.maximum(np.asarray(value, dtype=float), 0.0)


def lommel_seeliger_phase_function(phase_angle_rad: ArrayLike) -> FloatArray:
    """Lommel-Seeliger phase law with stable endpoint handling."""

    alpha = np.asarray(phase_angle_rad, dtype=float)
    epsilon = 1.0e-12
    alpha_eval = np.clip(alpha, epsilon, np.pi - epsilon)

    half = alpha_eval / 2.0
    value = (
        1.0
        + np.sin(half)
        * np.tan(half)
        * np.log(np.tan(alpha_eval / 4.0))
    )

    value = np.where(alpha <= epsilon, 1.0, value)
    value = np.where(alpha >= np.pi - epsilon, 0.0, value)

    return np.maximum(np.asarray(value, dtype=float), 0.0)


def stray_light_efficiency(off_axis_angle_rad: ArrayLike) -> FloatArray:
    """Residual-ghost stray-light efficiency.

    Interpolation is linear in log10(zeta) because the source curve is
    piecewise linear on a logarithmic vertical axis. The last value is held
    constant for angles >= 15 degrees.
    """

    theta_deg = np.rad2deg(np.asarray(off_axis_angle_rad, dtype=float))

    theta_points_deg = np.array([0.0, 5.0, 10.0, 15.0], dtype=float)
    zeta_points = np.array(
        [1.0, 2.45e-7, 5.16e-8, 2.45e-8],
        dtype=float,
    )

    theta_clipped_deg = np.clip(theta_deg, 0.0, 15.0)
    log10_zeta = np.interp(
        theta_clipped_deg,
        theta_points_deg,
        np.log10(zeta_points),
    )
    return np.asarray(10.0 ** log10_zeta, dtype=float)


def compute_apparent_angular_speed(
    observer_position_km: ArrayLike,
    observer_velocity_km_s: ArrayLike,
    asteroid_position_km: ArrayLike,
    asteroid_velocity_km_s: ArrayLike,
) -> FloatArray:
    """Calculate inertial line-of-sight angular speed in arcsec/s.

    The instantaneous angular speed is

        omega = ||rho x rho_dot|| / ||rho||^2

    where rho is the observer-to-asteroid relative position and rho_dot is the
    corresponding relative velocity. It is assumed constant over one
    exposure.
    """

    observer_position = _as_vector_array(
        observer_position_km,
        "observer_position_km",
    )
    observer_velocity = _as_vector_array(
        observer_velocity_km_s,
        "observer_velocity_km_s",
    )
    asteroid_position = _as_vector_array(
        asteroid_position_km,
        "asteroid_position_km",
    )
    asteroid_velocity = _as_vector_array(
        asteroid_velocity_km_s,
        "asteroid_velocity_km_s",
    )

    (
        observer_position,
        observer_velocity,
        asteroid_position,
        asteroid_velocity,
    ) = np.broadcast_arrays(
        observer_position,
        observer_velocity,
        asteroid_position,
        asteroid_velocity,
    )

    rho_km = asteroid_position - observer_position
    rho_dot_km_s = asteroid_velocity - observer_velocity

    range_squared_km2 = np.sum(rho_km * rho_km, axis=-1)
    if np.any(range_squared_km2 <= 0.0):
        raise ValueError("Observer-to-asteroid range must be positive.")

    angular_speed_rad_s = (
        np.linalg.norm(np.cross(rho_km, rho_dot_km_s), axis=-1)
        / range_squared_km2
    )

    return np.asarray(
        angular_speed_rad_s * ARCSEC_PER_RADIAN,
        dtype=float,
    )


def compute_asteroid_snr(
    payload: PayloadConfig,
    asteroid: AsteroidProperties,
    geometry: ObservationGeometry,
    environment: EnvironmentConfig | None = None,
    options: SNROptions | None = None,
    phase_model: HG12PhaseModel | None = None,
) -> SNRResult:
    """Compute the asteroid SNR for one observation or a broadcast batch.

    Parameters
    ----------
    payload
        Payload, detector, spectral-response, and zero-point configuration.
    asteroid
        Absolute magnitude, geometric albedo, and G12. These may be scalars or
        broadcastable batch arrays.
    geometry
        Cartesian positions and velocities. Vector quantities use shape
        (..., 3). All states must share one frame and epoch per batch element.
    environment
        Solar, Earth, and Moon constants. Visible defaults are used when
        omitted.
    options
        Model switches. Continuous effective aperture area is used by default.
    phase_model
        Optional custom HG12 spline model. The supplied table is used by
        default.

    Returns
    -------
    SNRResult
        SNR plus all major intermediate signal, noise, phase, aperture, and
        geometry quantities.

    Notes
    -----
    The calculation returns SNR only. It does not apply a detection threshold,
    field-of-view check, Earth/Moon occultation check, or final detection
    decision.
    """

    environment = environment or EnvironmentConfig()
    options = options or SNROptions()
    phase_model = phase_model or HG12PhaseModel.from_default_table()

    (
        absolute_magnitude,
        geometric_albedo,
        g12,
        psf_sigma_px,
        batch_geometry,
    ) = _prepare_batch_inputs(
        payload=payload,
        asteroid=asteroid,
        geometry=geometry,
    )

    if options.validate_inputs:
        _validate_model_inputs(
            payload=payload,
            environment=environment,
            options=options,
            absolute_magnitude=absolute_magnitude,
            geometric_albedo=geometric_albedo,
            g12=g12,
            psf_sigma_px=psf_sigma_px,
            geometry=batch_geometry,
        )

    wavelength_m, qe, throughput = _prepare_spectral_inputs(payload)

    zero_point_mag = compute_photometric_zero_point(
        payload=payload,
        wavelength_m=wavelength_m,
        quantum_efficiency=qe,
        optical_throughput=throughput,
    )

    solar_response_integral = compute_solar_response_integral(
        wavelength_m=wavelength_m,
        solar_temperature_k=environment.solar_temperature_k,
        quantum_efficiency=qe,
        optical_throughput=throughput,
    )

    geometry_terms = _compute_observation_geometry(batch_geometry)

    asteroid_diameter_m = diameter_from_absolute_magnitude(
        absolute_magnitude,
        geometric_albedo,
    )
    asteroid_radius_km = asteroid_diameter_m / 2000.0

    direct_phase = phase_model.evaluate(
        geometry_terms.direct_phase_angle_rad,
        g12,
    )
    earth_asteroid_phase = phase_model.evaluate(
        geometry_terms.earth_asteroid_phase_angle_rad,
        g12,
    )
    moon_asteroid_phase = phase_model.evaluate(
        geometry_terms.moon_asteroid_phase_angle_rad,
        g12,
    )

    apparent_magnitude = compute_apparent_magnitude(
        absolute_magnitude=absolute_magnitude,
        asteroid_sun_distance_au=(
            geometry_terms.asteroid_sun_distance_km / AU_KM
        ),
        asteroid_observer_distance_au=(
            geometry_terms.asteroid_observer_distance_km / AU_KM
        ),
        asteroid_phase_function=direct_phase.phase_function,
    )

    direct_rate_e_s = compute_direct_asteroid_rate(
        apparent_magnitude=apparent_magnitude,
        zero_point_mag=zero_point_mag,
    )

    zeros = np.zeros_like(absolute_magnitude, dtype=float)

    if options.include_earth_double_reflection:
        earth_double_reflection_rate_e_s = compute_double_reflection_rate(
            payload_area_m2=_payload_area_m2(payload.aperture_diameter_m),
            asteroid_albedo=geometric_albedo,
            asteroid_radius_km=asteroid_radius_km,
            asteroid_phase_function=earth_asteroid_phase.phase_function,
            body=environment.earth,
            body_phase_function=_evaluate_body_phase_law(
                environment.earth,
                geometry_terms.sun_earth_asteroid_phase_angle_rad,
            ),
            asteroid_observer_distance_km=(
                geometry_terms.asteroid_observer_distance_km
            ),
            asteroid_body_distance_km=(
                geometry_terms.asteroid_earth_distance_km
            ),
            body_sun_distance_km=geometry_terms.earth_sun_distance_km,
            asteroid_off_axis_angle_rad=(
                geometry_terms.asteroid_off_axis_angle_rad
            ),
            solar_radius_km=environment.solar_radius_km,
            solar_response_integral=solar_response_integral,
        )
    else:
        earth_double_reflection_rate_e_s = zeros.copy()

    if options.include_moon_double_reflection:
        moon_double_reflection_rate_e_s = compute_double_reflection_rate(
            payload_area_m2=_payload_area_m2(payload.aperture_diameter_m),
            asteroid_albedo=geometric_albedo,
            asteroid_radius_km=asteroid_radius_km,
            asteroid_phase_function=moon_asteroid_phase.phase_function,
            body=environment.moon,
            body_phase_function=_evaluate_body_phase_law(
                environment.moon,
                geometry_terms.sun_moon_asteroid_phase_angle_rad,
            ),
            asteroid_observer_distance_km=(
                geometry_terms.asteroid_observer_distance_km
            ),
            asteroid_body_distance_km=(
                geometry_terms.asteroid_moon_distance_km
            ),
            body_sun_distance_km=geometry_terms.moon_sun_distance_km,
            asteroid_off_axis_angle_rad=(
                geometry_terms.asteroid_off_axis_angle_rad
            ),
            solar_radius_km=environment.solar_radius_km,
            solar_response_integral=solar_response_integral,
        )
    else:
        moon_double_reflection_rate_e_s = zeros.copy()

    aperture = compute_streak_aperture(
        angular_rate_arcsec_s=(
            geometry_terms.apparent_angular_speed_arcsec_s
        ),
        exposure_time_s=payload.exposure_time_s,
        pixel_scale_arcsec_per_px=payload.pixel_scale_arcsec_per_px,
        psf_sigma_px=psf_sigma_px,
        pixel_mode=options.aperture_pixel_mode,
    )

    background_electrons = compute_background_electrons(
        n_pixels=aperture.n_pixels,
        exposure_time_s=payload.exposure_time_s,
        pixel_scale_arcsec_per_px=payload.pixel_scale_arcsec_per_px,
        background_surface_brightness_mag_arcsec2=(
            payload.background_surface_brightness_mag_arcsec2
        ),
        zero_point_mag=zero_point_mag,
    )
    dark_electrons = compute_dark_electrons(
        n_pixels=aperture.n_pixels,
        exposure_time_s=payload.exposure_time_s,
        dark_current_e_per_s_px=payload.dark_current_e_per_s_px,
    )
    read_noise_variance_e2 = compute_read_noise_variance(
        n_pixels=aperture.n_pixels,
        read_noise_e_rms_per_px=payload.read_noise_e_rms_per_px,
    )

    if options.include_earth_stray_light:
        earth_stray_light_rate_e_s_px = compute_stray_light_rate(
            payload=payload,
            payload_area_m2=_payload_area_m2(payload.aperture_diameter_m),
            body=environment.earth,
            body_phase_function=_evaluate_body_phase_law(
                environment.earth,
                geometry_terms.sun_earth_observer_phase_angle_rad,
            ),
            body_observer_distance_km=(
                geometry_terms.observer_earth_distance_km
            ),
            body_sun_distance_km=geometry_terms.earth_sun_distance_km,
            body_off_axis_angle_rad=(
                geometry_terms.earth_off_axis_angle_rad
            ),
            solar_radius_km=environment.solar_radius_km,
            solar_response_integral=solar_response_integral,
        )
    else:
        earth_stray_light_rate_e_s_px = zeros.copy()

    if options.include_moon_stray_light:
        moon_stray_light_rate_e_s_px = compute_stray_light_rate(
            payload=payload,
            payload_area_m2=_payload_area_m2(payload.aperture_diameter_m),
            body=environment.moon,
            body_phase_function=_evaluate_body_phase_law(
                environment.moon,
                geometry_terms.sun_moon_observer_phase_angle_rad,
            ),
            body_observer_distance_km=(
                geometry_terms.observer_moon_distance_km
            ),
            body_sun_distance_km=geometry_terms.moon_sun_distance_km,
            body_off_axis_angle_rad=(
                geometry_terms.moon_off_axis_angle_rad
            ),
            solar_radius_km=environment.solar_radius_km,
            solar_response_integral=solar_response_integral,
        )
    else:
        moon_stray_light_rate_e_s_px = zeros.copy()

    earth_stray_light_electrons = (
        aperture.n_pixels
        * payload.exposure_time_s
        * earth_stray_light_rate_e_s_px
    )
    moon_stray_light_electrons = (
        aperture.n_pixels
        * payload.exposure_time_s
        * moon_stray_light_rate_e_s_px
    )
    stray_light_electrons = (
        earth_stray_light_electrons
        + moon_stray_light_electrons
    )

    direct_signal_e = direct_rate_e_s * payload.exposure_time_s
    earth_double_reflection_e = (
        earth_double_reflection_rate_e_s
        * payload.exposure_time_s
    )
    moon_double_reflection_e = (
        moon_double_reflection_rate_e_s
        * payload.exposure_time_s
    )

    signal_electrons = (
        direct_signal_e
        + earth_double_reflection_e
        + moon_double_reflection_e
    )

    total_noise_variance_e2 = (
        signal_electrons
        + background_electrons
        + dark_electrons
        + read_noise_variance_e2
        + stray_light_electrons
    )
    total_noise_rms_e = np.sqrt(total_noise_variance_e2)

    snr = np.divide(
        signal_electrons,
        total_noise_rms_e,
        out=np.zeros_like(signal_electrons, dtype=float),
        where=total_noise_rms_e > 0.0,
    )

    return SNRResult(
        snr=np.asarray(snr, dtype=float),
        absolute_magnitude=absolute_magnitude,
        apparent_magnitude=apparent_magnitude,
        geometric_albedo=geometric_albedo,
        asteroid_diameter_m=asteroid_diameter_m,
        asteroid_radius_km=asteroid_radius_km,
        g12=g12,
        g1=direct_phase.g1,
        g2=direct_phase.g2,
        direct_phase_function=direct_phase.phase_function,
        earth_asteroid_phase_function=earth_asteroid_phase.phase_function,
        moon_asteroid_phase_function=moon_asteroid_phase.phase_function,
        zero_point_mag=float(zero_point_mag),
        wavelength_grid_m=wavelength_m,
        solar_response_integral=float(solar_response_integral),
        asteroid_direct_rate_e_s=direct_rate_e_s,
        earth_double_reflection_rate_e_s=(
            earth_double_reflection_rate_e_s
        ),
        moon_double_reflection_rate_e_s=(
            moon_double_reflection_rate_e_s
        ),
        direct_signal_e=direct_signal_e,
        earth_double_reflection_e=earth_double_reflection_e,
        moon_double_reflection_e=moon_double_reflection_e,
        signal_electrons=signal_electrons,
        background_electrons=background_electrons,
        dark_electrons=dark_electrons,
        read_noise_variance_e2=read_noise_variance_e2,
        earth_stray_light_rate_e_s_px=earth_stray_light_rate_e_s_px,
        moon_stray_light_rate_e_s_px=moon_stray_light_rate_e_s_px,
        earth_stray_light_electrons=earth_stray_light_electrons,
        moon_stray_light_electrons=moon_stray_light_electrons,
        stray_light_electrons=stray_light_electrons,
        source_shot_noise_variance_e2=signal_electrons,
        total_noise_variance_e2=total_noise_variance_e2,
        total_noise_rms_e=total_noise_rms_e,
        trail_length_px=aperture.trail_length_px,
        aperture_length_px=aperture.aperture_length_px,
        aperture_width_px=aperture.aperture_width_px,
        n_pixels=aperture.n_pixels,
        apparent_angular_speed_arcsec_s=(
            geometry_terms.apparent_angular_speed_arcsec_s
        ),
        asteroid_sun_distance_au=(
            geometry_terms.asteroid_sun_distance_km / AU_KM
        ),
        asteroid_observer_distance_au=(
            geometry_terms.asteroid_observer_distance_km / AU_KM
        ),
        direct_phase_angle_rad=geometry_terms.direct_phase_angle_rad,
        earth_asteroid_phase_angle_rad=(
            geometry_terms.earth_asteroid_phase_angle_rad
        ),
        moon_asteroid_phase_angle_rad=(
            geometry_terms.moon_asteroid_phase_angle_rad
        ),
        sun_earth_asteroid_phase_angle_rad=(
            geometry_terms.sun_earth_asteroid_phase_angle_rad
        ),
        sun_moon_asteroid_phase_angle_rad=(
            geometry_terms.sun_moon_asteroid_phase_angle_rad
        ),
        sun_earth_observer_phase_angle_rad=(
            geometry_terms.sun_earth_observer_phase_angle_rad
        ),
        sun_moon_observer_phase_angle_rad=(
            geometry_terms.sun_moon_observer_phase_angle_rad
        ),
        asteroid_off_axis_angle_rad=(
            geometry_terms.asteroid_off_axis_angle_rad
        ),
        earth_off_axis_angle_rad=geometry_terms.earth_off_axis_angle_rad,
        moon_off_axis_angle_rad=geometry_terms.moon_off_axis_angle_rad,
    )


def compute_apparent_magnitude(
    absolute_magnitude: ArrayLike,
    asteroid_sun_distance_au: ArrayLike,
    asteroid_observer_distance_au: ArrayLike,
    asteroid_phase_function: ArrayLike,
) -> FloatArray:
    """Calculate apparent visual magnitude.

    A zero phase function is treated as zero received direct signal and
    produces an infinite apparent magnitude.
    """

    h = np.asarray(absolute_magnitude, dtype=float)
    r_as_au = np.asarray(asteroid_sun_distance_au, dtype=float)
    r_ao_au = np.asarray(asteroid_observer_distance_au, dtype=float)
    phase = np.asarray(asteroid_phase_function, dtype=float)

    output = np.full(
        np.broadcast_shapes(
            h.shape,
            r_as_au.shape,
            r_ao_au.shape,
            phase.shape,
        ),
        np.inf,
        dtype=float,
    )

    h, r_as_au, r_ao_au, phase = np.broadcast_arrays(
        h,
        r_as_au,
        r_ao_au,
        phase,
    )

    valid = (
        (r_as_au > 0.0)
        & (r_ao_au > 0.0)
        & (phase > 0.0)
    )
    output[valid] = (
        h[valid]
        + 5.0 * np.log10(r_as_au[valid] * r_ao_au[valid])
        - 2.5 * np.log10(phase[valid])
    )
    return output


def compute_direct_asteroid_rate(
    apparent_magnitude: ArrayLike,
    zero_point_mag: float,
) -> FloatArray:
    """Convert apparent magnitude to direct source electron rate."""

    apparent_magnitude_array = np.asarray(
        apparent_magnitude,
        dtype=float,
    )

    rate = np.where(
        np.isfinite(apparent_magnitude_array),
        10.0 ** (
            -0.4
            * (
                apparent_magnitude_array
                - zero_point_mag
            )
        ),
        0.0,
    )
    return np.asarray(rate, dtype=float)


def compute_photometric_zero_point(
    payload: PayloadConfig,
    wavelength_m: FloatArray,
    quantum_efficiency: FloatArray,
    optical_throughput: FloatArray,
) -> float:
    """Return or calculate the system photometric zero point.

    Convention
    ----------
    The returned zero point is the magnitude of a source producing exactly
    1 electron per second in the complete payload system.

    If ``payload.zero_point_mag`` is supplied, it is returned directly.

    For scalar Vega flux density, the monochromatic photon-flux density at
    ``vega_reference_wavelength_m`` is assumed constant over the band. For a
    Vega flux array, the full wavelength-dependent integral is evaluated.
    """

    if payload.zero_point_mag is not None:
        zero_point = float(payload.zero_point_mag)
        if not np.isfinite(zero_point):
            raise ValueError("zero_point_mag must be finite.")
        return zero_point

    if payload.vega_flux_density_w_m2_m is None:
        raise ValueError(
            "Either zero_point_mag or vega_flux_density_w_m2_m is required."
        )

    area_m2 = _payload_area_m2(payload.aperture_diameter_m)
    response = quantum_efficiency * optical_throughput

    vega_flux = np.asarray(
        payload.vega_flux_density_w_m2_m,
        dtype=float,
    )

    if vega_flux.ndim == 0:
        if payload.vega_reference_wavelength_m is None:
            raise ValueError(
                "vega_reference_wavelength_m is required when the Vega "
                "flux density is supplied as a scalar."
            )

        photon_flux_density = (
            float(vega_flux)
            * payload.vega_reference_wavelength_m
            / (
                PLANCK_CONSTANT_J_S
                * SPEED_OF_LIGHT_M_S
            )
        )
        electron_rate_e_s = (
            area_m2
            * photon_flux_density
            * trapezoid(response, x=wavelength_m)
        )
    elif vega_flux.ndim == 1:
        if vega_flux.shape != wavelength_m.shape:
            raise ValueError(
                "The Vega flux-density array must have the same shape as "
                "wavelength_m."
            )

        vega_photon_flux_density = (
            vega_flux
            * wavelength_m
            / (
                PLANCK_CONSTANT_J_S
                * SPEED_OF_LIGHT_M_S
            )
        )
        electron_rate_e_s = (
            area_m2
            * trapezoid(
                vega_photon_flux_density * response,
                x=wavelength_m,
            )
        )
    else:
        raise ValueError(
            "vega_flux_density_w_m2_m must be a scalar or 1-D array."
        )

    if not np.isfinite(electron_rate_e_s) or electron_rate_e_s <= 0.0:
        raise ValueError(
            "The Vega-derived electron rate must be finite and positive."
        )

    return float(2.5 * np.log10(electron_rate_e_s))


def compute_solar_response_integral(
    wavelength_m: FloatArray,
    solar_temperature_k: float,
    quantum_efficiency: FloatArray,
    optical_throughput: FloatArray,
) -> float:
    """Evaluate the response-weighted solar photon integral.

    The returned quantity is

        integral p_gamma(lambda, T_s) QE(lambda) xi(lambda) d lambda

    using composite trapezoidal integration.
    """

    exponent = (
        PLANCK_CONSTANT_J_S
        * SPEED_OF_LIGHT_M_S
        / (
            wavelength_m
            * BOLTZMANN_CONSTANT_J_K
            * solar_temperature_k
        )
    )

    photon_spectral_radiance = (
        2.0
        * SPEED_OF_LIGHT_M_S
        / (
            wavelength_m**4
            * np.expm1(exponent)
        )
    )

    integrand = (
        photon_spectral_radiance
        * quantum_efficiency
        * optical_throughput
    )

    return float(trapezoid(integrand, x=wavelength_m))


def compute_double_reflection_rate(
    *,
    payload_area_m2: float,
    asteroid_albedo: ArrayLike,
    asteroid_radius_km: ArrayLike,
    asteroid_phase_function: ArrayLike,
    body: BodyProperties,
    body_phase_function: ArrayLike,
    asteroid_observer_distance_km: ArrayLike,
    asteroid_body_distance_km: ArrayLike,
    body_sun_distance_km: ArrayLike,
    asteroid_off_axis_angle_rad: ArrayLike,
    solar_radius_km: float,
    solar_response_integral: float,
) -> FloatArray:
    """Calculate Earth- or Moon-to-asteroid double-reflection electron rate."""

    asteroid_albedo = np.asarray(asteroid_albedo, dtype=float)
    asteroid_radius_m = (
        np.asarray(asteroid_radius_km, dtype=float)
        * 1000.0
    )
    asteroid_phase_function = np.asarray(
        asteroid_phase_function,
        dtype=float,
    )
    body_phase_function = np.asarray(body_phase_function, dtype=float)

    r_ao_m = (
        np.asarray(asteroid_observer_distance_km, dtype=float)
        * 1000.0
    )
    r_ab_m = (
        np.asarray(asteroid_body_distance_km, dtype=float)
        * 1000.0
    )
    r_bs_m = (
        np.asarray(body_sun_distance_km, dtype=float)
        * 1000.0
    )

    body_radius_m = body.radius_km * 1000.0
    solar_radius_m = solar_radius_km * 1000.0

    projected_cosine = np.maximum(
        np.cos(np.asarray(asteroid_off_axis_angle_rad, dtype=float)),
        0.0,
    )

    observer_collection_factor = (
        payload_area_m2
        * projected_cosine
        / r_ao_m**2
    )
    asteroid_reflection_factor = (
        asteroid_albedo
        * asteroid_phase_function
        * asteroid_radius_m**2
        / r_ab_m**2
    )
    body_reflection_factor = (
        body.geometric_albedo
        * body_phase_function
        * body_radius_m**2
    )
    solar_dilution_factor = (
        solar_radius_m**2
        / r_bs_m**2
    )
    spectral_electron_factor = (
        np.pi
        * solar_response_integral
    )

    rate = (
        observer_collection_factor
        * asteroid_reflection_factor
        * body_reflection_factor
        * solar_dilution_factor
        * spectral_electron_factor
    )
    return np.asarray(rate, dtype=float)


def compute_stray_light_rate(
    *,
    payload: PayloadConfig,
    payload_area_m2: float,
    body: BodyProperties,
    body_phase_function: ArrayLike,
    body_observer_distance_km: ArrayLike,
    body_sun_distance_km: ArrayLike,
    body_off_axis_angle_rad: ArrayLike,
    solar_radius_km: float,
    solar_response_integral: float,
) -> FloatArray:
    """Calculate body stray-light electron rate per aperture pixel.

    The adopted residual-ghost efficiency is wavelength independent, so it is
    factored outside the already-computed solar spectral integral.
    """

    r_ob_m = (
        np.asarray(body_observer_distance_km, dtype=float)
        * 1000.0
    )
    r_bs_m = (
        np.asarray(body_sun_distance_km, dtype=float)
        * 1000.0
    )
    body_phase_function = np.asarray(body_phase_function, dtype=float)
    theta_body = np.asarray(body_off_axis_angle_rad, dtype=float)

    projected_cosine = np.maximum(np.cos(theta_body), 0.0)
    zeta = stray_light_efficiency(theta_body)

    projected_pixel_area_m2 = (
        payload.pixel_pitch_m
        * r_ob_m
        / payload.focal_length_m
    ) ** 2

    solar_radius_m = solar_radius_km * 1000.0

    rate = (
        payload_area_m2
        * projected_cosine
        / (2.0 * np.pi * r_ob_m**2)
        * projected_pixel_area_m2
        * body.geometric_albedo
        * body_phase_function
        * (solar_radius_m**2 / r_bs_m**2)
        * np.pi
        * zeta
        * solar_response_integral
    )
    return np.asarray(rate, dtype=float)


def compute_streak_aperture(
    *,
    angular_rate_arcsec_s: ArrayLike,
    exposure_time_s: float,
    pixel_scale_arcsec_per_px: float,
    psf_sigma_px: ArrayLike,
    pixel_mode: Literal["continuous", "ceil"],
) -> ApertureTerms:
    """Calculate the rectangular aperture enclosing the full streaked PSF."""

    angular_rate = np.abs(
        np.asarray(angular_rate_arcsec_s, dtype=float)
    )
    psf_sigma = np.asarray(psf_sigma_px, dtype=float)

    trail_length_px = (
        angular_rate
        * exposure_time_s
        / pixel_scale_arcsec_per_px
    )
    aperture_length_px = trail_length_px + 6.0 * psf_sigma
    aperture_width_px = 6.0 * psf_sigma

    if pixel_mode == "ceil":
        aperture_length_px = np.ceil(aperture_length_px)
        aperture_width_px = np.ceil(aperture_width_px)
    elif pixel_mode != "continuous":
        raise ValueError(
            "aperture_pixel_mode must be 'continuous' or 'ceil'."
        )

    n_pixels = aperture_length_px * aperture_width_px

    return ApertureTerms(
        trail_length_px=np.asarray(trail_length_px, dtype=float),
        aperture_length_px=np.asarray(aperture_length_px, dtype=float),
        aperture_width_px=np.asarray(aperture_width_px, dtype=float),
        n_pixels=np.asarray(n_pixels, dtype=float),
    )


def compute_background_electrons(
    *,
    n_pixels: ArrayLike,
    exposure_time_s: float,
    pixel_scale_arcsec_per_px: float,
    background_surface_brightness_mag_arcsec2: float,
    zero_point_mag: float,
) -> FloatArray:
    """Calculate background electrons in the complete streak aperture."""

    return np.asarray(
        np.asarray(n_pixels, dtype=float)
        * exposure_time_s
        * pixel_scale_arcsec_per_px**2
        * 10.0 ** (
            -0.4
            * (
                background_surface_brightness_mag_arcsec2
                - zero_point_mag
            )
        ),
        dtype=float,
    )


def compute_dark_electrons(
    *,
    n_pixels: ArrayLike,
    exposure_time_s: float,
    dark_current_e_per_s_px: float,
) -> FloatArray:
    """Calculate dark-current electrons in the complete aperture."""

    return np.asarray(
        np.asarray(n_pixels, dtype=float)
        * exposure_time_s
        * dark_current_e_per_s_px,
        dtype=float,
    )


def compute_read_noise_variance(
    *,
    n_pixels: ArrayLike,
    read_noise_e_rms_per_px: float,
) -> FloatArray:
    """Calculate aperture read-noise variance in electron-squared units."""

    return np.asarray(
        np.asarray(n_pixels, dtype=float)
        * read_noise_e_rms_per_px**2,
        dtype=float,
    )


def _prepare_spectral_inputs(
    payload: PayloadConfig,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Create one wavelength grid and expand scalar spectral responses."""

    qe_raw = np.asarray(payload.quantum_efficiency, dtype=float)
    throughput_raw = np.asarray(payload.optical_throughput, dtype=float)

    qe_is_array = qe_raw.ndim > 0
    throughput_is_array = throughput_raw.ndim > 0

    vega_is_array = False
    if (
        payload.zero_point_mag is None
        and payload.vega_flux_density_w_m2_m is not None
    ):
        vega_raw = np.asarray(
            payload.vega_flux_density_w_m2_m,
            dtype=float,
        )
        if vega_raw.ndim > 1:
            raise ValueError(
                "vega_flux_density_w_m2_m must be a scalar or 1-D array."
            )
        vega_is_array = vega_raw.ndim > 0

    if qe_raw.ndim > 1:
        raise ValueError("quantum_efficiency must be a scalar or 1-D array.")
    if throughput_raw.ndim > 1:
        raise ValueError(
            "optical_throughput must be a scalar or 1-D array."
        )

    if payload.wavelength_m is not None:
        wavelength_m = np.asarray(payload.wavelength_m, dtype=float)
        if wavelength_m.ndim != 1:
            raise ValueError("wavelength_m must be one-dimensional.")
    else:
        if qe_is_array or throughput_is_array or vega_is_array:
            raise ValueError(
                "wavelength_m is required when QE, throughput, or the Vega "
                "flux density is supplied as an array."
            )
        if (
            payload.wavelength_lower_m is None
            or payload.wavelength_upper_m is None
        ):
            raise ValueError(
                "Provide wavelength_m or both wavelength_lower_m and "
                "wavelength_upper_m."
            )
        if payload.spectral_samples < 2:
            raise ValueError("spectral_samples must be at least 2.")

        wavelength_m = np.linspace(
            payload.wavelength_lower_m,
            payload.wavelength_upper_m,
            payload.spectral_samples,
            dtype=float,
        )

    if qe_raw.ndim == 0:
        qe = np.full_like(wavelength_m, float(qe_raw))
    else:
        if qe_raw.shape != wavelength_m.shape:
            raise ValueError(
                "The QE array must have the same shape as wavelength_m."
            )
        qe = np.asarray(qe_raw, dtype=float)

    if throughput_raw.ndim == 0:
        throughput = np.full_like(
            wavelength_m,
            float(throughput_raw),
        )
    else:
        if throughput_raw.shape != wavelength_m.shape:
            raise ValueError(
                "The throughput array must have the same shape as wavelength_m."
            )
        throughput = np.asarray(throughput_raw, dtype=float)

    if np.any(~np.isfinite(wavelength_m)):
        raise ValueError("wavelength_m must contain only finite values.")
    if np.any(wavelength_m <= 0.0):
        raise ValueError("All wavelengths must be positive.")
    if np.any(np.diff(wavelength_m) <= 0.0):
        raise ValueError("wavelength_m must be strictly increasing.")

    if np.any(~np.isfinite(qe)) or np.any((qe < 0.0) | (qe > 1.0)):
        raise ValueError("Quantum efficiency must lie in [0, 1].")
    if (
        np.any(~np.isfinite(throughput))
        or np.any((throughput < 0.0) | (throughput > 1.0))
    ):
        raise ValueError("Optical throughput must lie in [0, 1].")

    return wavelength_m, qe, throughput


def _prepare_batch_inputs(
    *,
    payload: PayloadConfig,
    asteroid: AsteroidProperties,
    geometry: ObservationGeometry,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    ObservationGeometry,
]:
    """Broadcast all observation and asteroid inputs to one batch shape."""

    vector_fields = {
        "observer_position_km": _as_vector_array(
            geometry.observer_position_km,
            "observer_position_km",
        ),
        "observer_velocity_km_s": _as_vector_array(
            geometry.observer_velocity_km_s,
            "observer_velocity_km_s",
        ),
        "asteroid_position_km": _as_vector_array(
            geometry.asteroid_position_km,
            "asteroid_position_km",
        ),
        "asteroid_velocity_km_s": _as_vector_array(
            geometry.asteroid_velocity_km_s,
            "asteroid_velocity_km_s",
        ),
        "sun_position_km": _as_vector_array(
            geometry.sun_position_km,
            "sun_position_km",
        ),
        "earth_position_km": _as_vector_array(
            geometry.earth_position_km,
            "earth_position_km",
        ),
        "moon_position_km": _as_vector_array(
            geometry.moon_position_km,
            "moon_position_km",
        ),
        "boresight_unit_vector": _as_vector_array(
            geometry.boresight_unit_vector,
            "boresight_unit_vector",
        ),
    }

    absolute_magnitude = np.asarray(
        asteroid.absolute_magnitude,
        dtype=float,
    )
    geometric_albedo = np.asarray(
        asteroid.geometric_albedo,
        dtype=float,
    )
    g12 = np.asarray(asteroid.g12, dtype=float)
    psf_sigma_px = np.asarray(payload.psf_sigma_px, dtype=float)

    scalar_arrays = [
        absolute_magnitude,
        geometric_albedo,
        g12,
        psf_sigma_px,
    ]

    if geometry.asteroid_angular_rate_arcsec_s is not None:
        angular_rate = np.asarray(
            geometry.asteroid_angular_rate_arcsec_s,
            dtype=float,
        )
        scalar_arrays.append(angular_rate)
    else:
        angular_rate = None

    batch_shapes = [
        array.shape[:-1]
        for array in vector_fields.values()
    ]
    batch_shapes.extend(array.shape for array in scalar_arrays)
    batch_shape = np.broadcast_shapes(*batch_shapes)

    broadcast_vectors = {
        name: np.broadcast_to(
            array,
            batch_shape + (3,),
        ).astype(float, copy=False)
        for name, array in vector_fields.items()
    }

    absolute_magnitude = np.broadcast_to(
        absolute_magnitude,
        batch_shape,
    ).astype(float, copy=False)
    geometric_albedo = np.broadcast_to(
        geometric_albedo,
        batch_shape,
    ).astype(float, copy=False)
    g12 = np.broadcast_to(
        g12,
        batch_shape,
    ).astype(float, copy=False)
    psf_sigma_px = np.broadcast_to(
        psf_sigma_px,
        batch_shape,
    ).astype(float, copy=False)

    if angular_rate is not None:
        angular_rate = np.broadcast_to(
            angular_rate,
            batch_shape,
        ).astype(float, copy=False)

    batch_geometry = ObservationGeometry(
        observer_position_km=broadcast_vectors["observer_position_km"],
        observer_velocity_km_s=broadcast_vectors["observer_velocity_km_s"],
        asteroid_position_km=broadcast_vectors["asteroid_position_km"],
        asteroid_velocity_km_s=broadcast_vectors["asteroid_velocity_km_s"],
        sun_position_km=broadcast_vectors["sun_position_km"],
        earth_position_km=broadcast_vectors["earth_position_km"],
        moon_position_km=broadcast_vectors["moon_position_km"],
        boresight_unit_vector=broadcast_vectors["boresight_unit_vector"],
        asteroid_angular_rate_arcsec_s=angular_rate,
    )

    return (
        absolute_magnitude,
        geometric_albedo,
        g12,
        psf_sigma_px,
        batch_geometry,
    )


def _compute_observation_geometry(
    geometry: ObservationGeometry,
) -> GeometryTerms:
    """Derive all required distances, phase angles, and off-axis angles."""

    observer = np.asarray(geometry.observer_position_km, dtype=float)
    asteroid = np.asarray(geometry.asteroid_position_km, dtype=float)
    sun = np.asarray(geometry.sun_position_km, dtype=float)
    earth = np.asarray(geometry.earth_position_km, dtype=float)
    moon = np.asarray(geometry.moon_position_km, dtype=float)

    boresight = _normalize_vectors(
        np.asarray(geometry.boresight_unit_vector, dtype=float),
        "boresight_unit_vector",
    )

    asteroid_to_sun = sun - asteroid
    asteroid_to_observer = observer - asteroid
    asteroid_to_earth = earth - asteroid
    asteroid_to_moon = moon - asteroid

    observer_to_asteroid = asteroid - observer
    observer_to_earth = earth - observer
    observer_to_moon = moon - observer

    earth_to_sun = sun - earth
    earth_to_asteroid = asteroid - earth
    earth_to_observer = observer - earth

    moon_to_sun = sun - moon
    moon_to_asteroid = asteroid - moon
    moon_to_observer = observer - moon

    if geometry.asteroid_angular_rate_arcsec_s is None:
        angular_speed_arcsec_s = compute_apparent_angular_speed(
            observer_position_km=observer,
            observer_velocity_km_s=geometry.observer_velocity_km_s,
            asteroid_position_km=asteroid,
            asteroid_velocity_km_s=geometry.asteroid_velocity_km_s,
        )
    else:
        angular_speed_arcsec_s = np.abs(
            np.asarray(
                geometry.asteroid_angular_rate_arcsec_s,
                dtype=float,
            )
        )

    return GeometryTerms(
        asteroid_sun_distance_km=_vector_norm(asteroid_to_sun),
        asteroid_observer_distance_km=_vector_norm(
            asteroid_to_observer
        ),
        asteroid_earth_distance_km=_vector_norm(asteroid_to_earth),
        asteroid_moon_distance_km=_vector_norm(asteroid_to_moon),
        earth_sun_distance_km=_vector_norm(earth_to_sun),
        moon_sun_distance_km=_vector_norm(moon_to_sun),
        observer_earth_distance_km=_vector_norm(observer_to_earth),
        observer_moon_distance_km=_vector_norm(observer_to_moon),
        direct_phase_angle_rad=_angle_between(
            asteroid_to_observer,
            asteroid_to_sun,
        ),
        earth_asteroid_phase_angle_rad=_angle_between(
            asteroid_to_earth,
            asteroid_to_observer,
        ),
        moon_asteroid_phase_angle_rad=_angle_between(
            asteroid_to_moon,
            asteroid_to_observer,
        ),
        sun_earth_asteroid_phase_angle_rad=_angle_between(
            earth_to_sun,
            earth_to_asteroid,
        ),
        sun_moon_asteroid_phase_angle_rad=_angle_between(
            moon_to_sun,
            moon_to_asteroid,
        ),
        sun_earth_observer_phase_angle_rad=_angle_between(
            earth_to_sun,
            earth_to_observer,
        ),
        sun_moon_observer_phase_angle_rad=_angle_between(
            moon_to_sun,
            moon_to_observer,
        ),
        asteroid_off_axis_angle_rad=_angle_between(
            boresight,
            observer_to_asteroid,
        ),
        earth_off_axis_angle_rad=_angle_between(
            boresight,
            observer_to_earth,
        ),
        moon_off_axis_angle_rad=_angle_between(
            boresight,
            observer_to_moon,
        ),
        apparent_angular_speed_arcsec_s=np.asarray(
            angular_speed_arcsec_s,
            dtype=float,
        ),
    )


def _validate_model_inputs(
    *,
    payload: PayloadConfig,
    environment: EnvironmentConfig,
    options: SNROptions,
    absolute_magnitude: FloatArray,
    geometric_albedo: FloatArray,
    g12: FloatArray,
    psf_sigma_px: FloatArray,
    geometry: ObservationGeometry,
) -> None:
    """Validate scalar, batch, and physical input ranges."""

    positive_payload_fields = {
        "exposure_time_s": payload.exposure_time_s,
        "aperture_diameter_m": payload.aperture_diameter_m,
        "focal_length_m": payload.focal_length_m,
        "pixel_scale_arcsec_per_px": payload.pixel_scale_arcsec_per_px,
        "pixel_pitch_m": payload.pixel_pitch_m,
    }
    for name, value in positive_payload_fields.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    if payload.dark_current_e_per_s_px < 0.0:
        raise ValueError("dark_current_e_per_s_px must be nonnegative.")
    if payload.read_noise_e_rms_per_px < 0.0:
        raise ValueError("read_noise_e_rms_per_px must be nonnegative.")
    if not np.isfinite(
        payload.background_surface_brightness_mag_arcsec2
    ):
        raise ValueError(
            "background_surface_brightness_mag_arcsec2 must be finite."
        )

    for name, array in {
        "absolute_magnitude": absolute_magnitude,
        "geometric_albedo": geometric_albedo,
        "g12": g12,
        "psf_sigma_px": psf_sigma_px,
    }.items():
        if np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")

    if np.any(geometric_albedo <= 0.0):
        raise ValueError("geometric_albedo must be positive.")
    if np.any((g12 < 0.0) | (g12 > 1.0)):
        raise ValueError("g12 must lie in [0, 1].")
    if np.any(psf_sigma_px <= 0.0):
        raise ValueError("psf_sigma_px must be positive.")

    for name in (
        "observer_position_km",
        "observer_velocity_km_s",
        "asteroid_position_km",
        "asteroid_velocity_km_s",
        "sun_position_km",
        "earth_position_km",
        "moon_position_km",
        "boresight_unit_vector",
    ):
        array = np.asarray(getattr(geometry, name), dtype=float)
        if np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")

    if geometry.asteroid_angular_rate_arcsec_s is not None:
        angular_rate = np.asarray(
            geometry.asteroid_angular_rate_arcsec_s,
            dtype=float,
        )
        if np.any(~np.isfinite(angular_rate)):
            raise ValueError(
                "asteroid_angular_rate_arcsec_s must be finite."
            )

    boresight_norm = _vector_norm(
        np.asarray(geometry.boresight_unit_vector, dtype=float)
    )
    if np.any(boresight_norm <= 0.0):
        raise ValueError("boresight_unit_vector must be nonzero.")

    if environment.solar_radius_km <= 0.0:
        raise ValueError("solar_radius_km must be positive.")
    if environment.solar_temperature_k <= 0.0:
        raise ValueError("solar_temperature_k must be positive.")

    for body_name, body in (
        ("earth", environment.earth),
        ("moon", environment.moon),
    ):
        if body.radius_km <= 0.0:
            raise ValueError(f"{body_name}.radius_km must be positive.")
        if not 0.0 <= body.geometric_albedo <= 1.0:
            raise ValueError(
                f"{body_name}.geometric_albedo must lie in [0, 1]."
            )

    if options.aperture_pixel_mode not in ("continuous", "ceil"):
        raise ValueError(
            "aperture_pixel_mode must be 'continuous' or 'ceil'."
        )


def _evaluate_body_phase_law(
    body: BodyProperties,
    phase_angle_rad: ArrayLike,
) -> FloatArray:
    if body.phase_law == "lambert":
        return lambert_phase_function(phase_angle_rad)
    if body.phase_law == "lommel_seeliger":
        return lommel_seeliger_phase_function(phase_angle_rad)
    raise ValueError(f"Unsupported phase law: {body.phase_law!r}")


def _payload_area_m2(aperture_diameter_m: float) -> float:
    return float(np.pi * aperture_diameter_m**2 / 4.0)


def _as_vector_array(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3).")
    return array


def _vector_norm(vectors: ArrayLike) -> FloatArray:
    return np.asarray(
        np.linalg.norm(np.asarray(vectors, dtype=float), axis=-1),
        dtype=float,
    )


def _normalize_vectors(
    vectors: ArrayLike,
    name: str,
) -> FloatArray:
    vectors_array = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors_array, axis=-1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} must contain nonzero vectors.")
    return np.asarray(vectors_array / norms, dtype=float)


def _angle_between(
    vector_1: ArrayLike,
    vector_2: ArrayLike,
) -> FloatArray:
    v1 = np.asarray(vector_1, dtype=float)
    v2 = np.asarray(vector_2, dtype=float)

    norm_1 = np.linalg.norm(v1, axis=-1)
    norm_2 = np.linalg.norm(v2, axis=-1)

    if np.any(norm_1 <= 0.0) or np.any(norm_2 <= 0.0):
        raise ValueError(
            "Cannot calculate an angle involving a zero-length vector."
        )

    cosine = np.sum(v1 * v2, axis=-1) / (norm_1 * norm_2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.asarray(np.arccos(cosine), dtype=float)


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example with three observation epochs. All Cartesian states are expressed
    # in one Earth-centred inertial frame for illustration. The origin is
    # arbitrary for this model, provided every position uses the same frame.


    payload = PayloadConfig(
        exposure_time_s=30.0,
        aperture_diameter_m=0.10,
        focal_length_m=0.30,
        pixel_scale_arcsec_per_px=10.0,
        pixel_pitch_m=15.0e-6,
        psf_sigma_px=1.2,

        # Scalars mean constant response over the complete band.
        quantum_efficiency=0.80,
        optical_throughput=0.70,

        dark_current_e_per_s_px=0.01,
        read_noise_e_rms_per_px=2.0,
        background_surface_brightness_mag_arcsec2=22.0,

        wavelength_lower_m=400.0e-9,
        wavelength_upper_m=800.0e-9,
        spectral_samples=2001,

        # The zero point is the magnitude producing 1 electron per second.
        zero_point_mag=23.0,
    )

    asteroid = AsteroidProperties(
        absolute_magnitude=30.0,
        geometric_albedo=0.14,
        g12=0.40,
    )

    observer_position_km = np.array(
        [
            [-1.50e6, 0.0, 0.0],
            [-1.50e6, 0.0, 0.0],
            [-1.50e6, 0.0, 0.0],
        ]
    )
    observer_velocity_km_s = np.array(
        [
            [0.0, -0.20, 0.0],
            [0.0, -0.20, 0.0],
            [0.0, -0.20, 0.0],
        ]
    )

    asteroid_position_km = np.array(
        [
            [-1.20e6, 20_000.0, 5_000.0],
            [-1.18e6, 26_000.0, 5_500.0],
            [-1.16e6, 33_000.0, 6_000.0],
        ]
    )
    asteroid_velocity_km_s = np.array(
        [
            [0.15, 0.45, 0.02],
            [0.15, 0.45, 0.02],
            [0.15, 0.45, 0.02],
        ]
    )

    sun_position_km = np.array([-AU_KM, 0.0, 0.0])
    earth_position_km = np.array([0.0, 0.0, 0.0])
    moon_position_km = np.array([384_400.0, 0.0, 0.0])

    # Point the payload at the asteroid at each observation epoch. Each
    # boresight is treated as fixed during its corresponding exposure.
    boresight = asteroid_position_km - observer_position_km
    boresight /= np.linalg.norm(boresight, axis=-1, keepdims=True)

    geometry = ObservationGeometry(
        observer_position_km=observer_position_km,
        observer_velocity_km_s=observer_velocity_km_s,
        asteroid_position_km=asteroid_position_km,
        asteroid_velocity_km_s=asteroid_velocity_km_s,
        sun_position_km=sun_position_km,
        earth_position_km=earth_position_km,
        moon_position_km=moon_position_km,
        boresight_unit_vector=boresight,
    )

    result = compute_asteroid_snr(
        payload=payload,
        asteroid=asteroid,
        geometry=geometry,
        environment=EnvironmentConfig(),
        options=SNROptions(
            aperture_pixel_mode="continuous",
        ),
    )

    print("Asteroid SNR example")
    print("--------------------")
    for index in range(result.snr.size):
        print(
            f"Epoch {index}: "
            f"SNR={result.snr[index]:.6g}, "
            f"V={result.apparent_magnitude[index]:.3f}, "
            f"omega={result.apparent_angular_speed_arcsec_s[index]:.3f} "
            f"arcsec/s, "
            f"trail={result.trail_length_px[index]:.3f} px, "
            f"signal={result.signal_electrons[index]:.3e} e-, "
            f"noise RMS={result.total_noise_rms_e[index]:.3e} e-"
        )
