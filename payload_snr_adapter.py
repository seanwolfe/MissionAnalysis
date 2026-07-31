from __future__ import annotations

"""Mission-facing adapter for the pure asteroid SNR physics model.

The adapter is responsible for:
- loading and validating the YAML/NPZ payload configuration;
- constructing model dataclasses;
- preparing payload-static spectral quantities once;
- accepting Geo-EME state arrays from the overall simulation;
- enforcing state-derived apparent angular rate; and
- applying the configured SNR threshold.

It deliberately does not perform FOV, occultation, EMS-exclusion, IOD, or OD
logic. Those remain in the overall mission simulation.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from payload_config_loader import LoadedPayloadConfiguration, load_payload_configuration
from payload_asteroid_snr_model import (
    AsteroidProperties,
    BodyProperties,
    EnvironmentConfig,
    HG12PhaseModel,
    ObservationGeometry,
    PayloadConfig,
    PreparedPayloadTerms,
    SNRResult,
    SNROptions,
    StrayLightConfig,
    compute_apparent_angular_speed,
    compute_asteroid_snr,
    prepare_payload_terms,
)


BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SNREvaluation:
    """Physics result together with the configured threshold decision."""

    result: SNRResult
    pass_mask: BoolArray


@dataclass(frozen=True)
class PayloadSNREvaluator:
    """Prepared, reusable evaluator built from the overall YAML configuration."""

    enabled: bool
    snr_threshold: float
    absolute_magnitude_metadata_column: str
    asteroid_geometric_albedo: float
    asteroid_g12: float

    payload: PayloadConfig
    environment: EnvironmentConfig
    options: SNROptions
    stray_light: StrayLightConfig
    phase_model: HG12PhaseModel
    prepared_terms: PreparedPayloadTerms
    spectral_file_path: Path

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        config_path: str | Path | None = None,
        base_dir: str | Path | None = None,
    ) -> "PayloadSNREvaluator":
        """Build an evaluator from an already-loaded overall configuration."""

        loaded = load_payload_configuration(
            config,
            config_path=config_path,
            base_dir=base_dir,
        )
        return cls._from_loaded_configuration(loaded)

    @classmethod
    def _from_loaded_configuration(
        cls,
        loaded: LoadedPayloadConfiguration,
    ) -> "PayloadSNREvaluator":
        cfg = loaded.payload_snr
        optics = cfg["optics"]
        detector = cfg["detector"]
        spectral = cfg["spectral_response"]
        asteroid_cfg = cfg["asteroid"]
        environment_cfg = cfg["environment"]
        options_cfg = cfg["options"]
        stray_cfg = cfg["stray_light"]

        payload = PayloadConfig(
            exposure_time_s=float(detector["exposure_time_s"]),
            aperture_diameter_m=float(optics["aperture_diameter_m"]),
            focal_length_m=float(optics["focal_length_m"]),
            pixel_scale_arcsec_per_px=float(optics["pixel_scale_arcsec_per_px"]),
            pixel_pitch_m=float(optics["pixel_pitch_m"]),
            psf_sigma_px=float(optics["psf_sigma_px"]),
            quantum_efficiency=loaded.quantum_efficiency.copy(),
            optical_throughput=loaded.optical_throughput.copy(),
            dark_current_e_per_s_px=float(detector["dark_current_e_per_s_px"]),
            read_noise_e_rms_per_px=float(detector["read_noise_e_rms_per_px"]),
            background_surface_brightness_mag_arcsec2=float(
                detector["background_surface_brightness_mag_arcsec2"]
            ),
            wavelength_m=loaded.wavelength_m.copy(),
            wavelength_lower_m=_optional_float(spectral.get("wavelength_lower_m")),
            wavelength_upper_m=_optional_float(spectral.get("wavelength_upper_m")),
            spectral_samples=int(spectral.get("spectral_samples", 2001)),
            zero_point_mag=_optional_float(spectral.get("zero_point_mag")),
            vega_flux_density_w_m2_m=_optional_float(
                spectral.get("vega_flux_density_w_m2_m")
            ),
            vega_reference_wavelength_m=_optional_float(
                spectral.get("vega_reference_wavelength_m")
            ),
        )

        environment = EnvironmentConfig(
            astronomical_unit_km=float(
                environment_cfg["astronomical_unit_km"]
            ),
            solar_radius_km=float(environment_cfg["solar_radius_km"]),
            solar_temperature_k=float(environment_cfg["solar_temperature_k"]),
            earth=_body_from_mapping(environment_cfg["earth"]),
            moon=_body_from_mapping(environment_cfg["moon"]),
        )

        options = SNROptions(
            include_earth_double_reflection=bool(
                options_cfg["include_earth_double_reflection"]
            ),
            include_moon_double_reflection=bool(
                options_cfg["include_moon_double_reflection"]
            ),
            include_earth_stray_light=bool(
                options_cfg["include_earth_stray_light"]
            ),
            include_moon_stray_light=bool(
                options_cfg["include_moon_stray_light"]
            ),
            aperture_pixel_mode=str(options_cfg["aperture_pixel_mode"]),
            validate_inputs=bool(options_cfg["validate_inputs"]),
        )

        stray_light = StrayLightConfig(
            off_axis_angle_deg=np.asarray(
                stray_cfg["off_axis_angle_deg"], dtype=float
            ).copy(),
            efficiency=np.asarray(stray_cfg["efficiency"], dtype=float).copy(),
            interpolation=str(stray_cfg["interpolation"]),
            hold_last_value=bool(stray_cfg["hold_last_value"]),
            zero_at_angle_deg=_optional_float(
                stray_cfg.get("zero_at_angle_deg")
            ),
        )

        phase_model = HG12PhaseModel.from_default_table()
        prepared_terms = prepare_payload_terms(payload, environment)

        return cls(
            enabled=bool(cfg["enabled"]),
            snr_threshold=float(cfg["snr_threshold"]),
            absolute_magnitude_metadata_column=str(
                asteroid_cfg["absolute_magnitude_metadata_column"]
            ),
            asteroid_geometric_albedo=float(asteroid_cfg["geometric_albedo"]),
            asteroid_g12=float(asteroid_cfg["g12"]),
            payload=payload,
            environment=environment,
            options=options,
            stray_light=stray_light,
            phase_model=phase_model,
            prepared_terms=prepared_terms,
            spectral_file_path=loaded.spectral_file_path,
        )

    @property
    def geometric_albedo(self) -> float:
        """Configured asteroid geometric albedo."""

        return float(self.asteroid_geometric_albedo)

    @property
    def g12(self) -> float:
        """Configured asteroid G12 slope parameter."""

        return float(self.asteroid_g12)

    def absolute_magnitude_from_metadata(self, metadata: Any) -> float:
        """Read the configured H field from a mapping or pandas-like row."""

        column = self.absolute_magnitude_metadata_column
        try:
            value = metadata[column]
        except Exception as exc:
            raise KeyError(
                f"Asteroid metadata does not contain configured H column {column!r}."
            ) from exc
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Absolute magnitude {column!r} must be finite.")
        return value

    def evaluate_batch(
        self,
        *,
        absolute_magnitude: ArrayLike,
        asteroid_position_km: ArrayLike,
        asteroid_velocity_km_s: ArrayLike,
        observer_position_km: ArrayLike,
        observer_velocity_km_s: ArrayLike,
        sun_position_km: ArrayLike,
        earth_position_km: ArrayLike,
        moon_position_km: ArrayLike,
        boresight_unit_vector: ArrayLike,
    ) -> SNREvaluation:
        """Evaluate one observation or a broadcast batch in a common frame.

        The production adapter always leaves the angular-rate override unset,
        so apparent angular speed is derived from relative position and
        velocity states.
        """

        asteroid = AsteroidProperties(
            absolute_magnitude=absolute_magnitude,
            geometric_albedo=self.asteroid_geometric_albedo,
            g12=self.asteroid_g12,
        )
        geometry = ObservationGeometry(
            observer_position_km=observer_position_km,
            observer_velocity_km_s=observer_velocity_km_s,
            asteroid_position_km=asteroid_position_km,
            asteroid_velocity_km_s=asteroid_velocity_km_s,
            sun_position_km=sun_position_km,
            earth_position_km=earth_position_km,
            moon_position_km=moon_position_km,
            boresight_unit_vector=boresight_unit_vector,
            asteroid_angular_rate_arcsec_s=None,
        )

        result = compute_asteroid_snr(
            payload=self.payload,
            asteroid=asteroid,
            geometry=geometry,
            environment=self.environment,
            options=self.options,
            phase_model=self.phase_model,
            stray_light=self.stray_light,
            prepared_terms=self.prepared_terms,
        )
        pass_mask = np.asarray(result.snr >= self.snr_threshold, dtype=bool)
        return SNREvaluation(result=result, pass_mask=pass_mask)

    def evaluate_from_metadata(self, metadata: Any, **geometry_kwargs: Any) -> SNREvaluation:
        """Evaluate using H read from the configured metadata column."""

        return self.evaluate_batch(
            absolute_magnitude=self.absolute_magnitude_from_metadata(metadata),
            **geometry_kwargs,
        )

    def evaluate_batch_chunked(
        self,
        *,
        chunk_size: int,
        absolute_magnitude: ArrayLike,
        asteroid_position_km: ArrayLike,
        asteroid_velocity_km_s: ArrayLike,
        observer_position_km: ArrayLike,
        observer_velocity_km_s: ArrayLike,
        sun_position_km: ArrayLike,
        earth_position_km: ArrayLike,
        moon_position_km: ArrayLike,
        boresight_unit_vector: ArrayLike,
    ) -> SNREvaluation:
        """Evaluate a leading-dimension batch in bounded-size chunks."""

        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive.")
        asteroid_positions = np.asarray(asteroid_position_km, dtype=float)
        if asteroid_positions.ndim != 2 or asteroid_positions.shape[1] != 3:
            raise ValueError(
                "Chunked evaluation requires asteroid_position_km with shape (N, 3)."
            )
        count = asteroid_positions.shape[0]
        if count == 0:
            raise ValueError("Chunked evaluation requires at least one observation.")

        inputs = {
            "absolute_magnitude": absolute_magnitude,
            "asteroid_position_km": asteroid_position_km,
            "asteroid_velocity_km_s": asteroid_velocity_km_s,
            "observer_position_km": observer_position_km,
            "observer_velocity_km_s": observer_velocity_km_s,
            "sun_position_km": sun_position_km,
            "earth_position_km": earth_position_km,
            "moon_position_km": moon_position_km,
            "boresight_unit_vector": boresight_unit_vector,
        }

        evaluations: list[SNREvaluation] = []
        for start in range(0, count, int(chunk_size)):
            stop = min(start + int(chunk_size), count)
            chunk_inputs = {
                key: _slice_input(value, start, stop, count, vector=key != "absolute_magnitude")
                for key, value in inputs.items()
            }
            evaluations.append(self.evaluate_batch(**chunk_inputs))

        result = _concatenate_snr_results([item.result for item in evaluations])
        pass_mask = np.concatenate(
            [np.atleast_1d(item.pass_mask) for item in evaluations], axis=0
        ).astype(bool, copy=False)
        return SNREvaluation(result=result, pass_mask=pass_mask)

    def expected_angular_speed_arcsec_s(
        self,
        *,
        observer_position_km: ArrayLike,
        observer_velocity_km_s: ArrayLike,
        asteroid_position_km: ArrayLike,
        asteroid_velocity_km_s: ArrayLike,
    ) -> np.ndarray:
        """Expose the exact state-derived angular-rate calculation for checks."""

        return compute_apparent_angular_speed(
            observer_position_km=observer_position_km,
            observer_velocity_km_s=observer_velocity_km_s,
            asteroid_position_km=asteroid_position_km,
            asteroid_velocity_km_s=asteroid_velocity_km_s,
        )



def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _body_from_mapping(mapping: Mapping[str, Any]) -> BodyProperties:
    return BodyProperties(
        radius_km=float(mapping["radius_km"]),
        geometric_albedo=float(mapping["geometric_albedo"]),
        phase_law=str(mapping["phase_law"]),
    )


def _slice_input(
    value: ArrayLike,
    start: int,
    stop: int,
    total: int,
    *,
    vector: bool,
) -> ArrayLike:
    array = np.asarray(value)
    if array.ndim == 0:
        return value
    if vector and array.shape == (3,):
        return value
    if array.shape[0] == total:
        return array[start:stop]
    return value


def _concatenate_snr_results(results: list[SNRResult]) -> SNRResult:
    if not results:
        raise ValueError("At least one SNRResult is required.")

    static_fields = {
        "zero_point_mag",
        "wavelength_grid_m",
        "solar_response_integral",
    }
    values: dict[str, Any] = {}
    first = results[0]
    for field in fields(SNRResult):
        name = field.name
        if name in static_fields:
            values[name] = getattr(first, name)
        else:
            values[name] = np.concatenate(
                [np.atleast_1d(getattr(result, name)) for result in results],
                axis=0,
            )
    return SNRResult(**values)
