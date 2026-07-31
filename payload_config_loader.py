from __future__ import annotations

"""Load and validate the Stage-2 payload SNR configuration.

This module performs configuration and spectral-data validation only. It does
not calculate SNR and does not alter overall-simulation detection behaviour.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from simulation_config import (
    load_simulation_config,
    normalize_simulation_config,
)


@dataclass(frozen=True)
class LoadedPayloadConfiguration:
    """Validated payload configuration and spectral response arrays."""

    payload_snr: dict[str, Any]
    spectral_file_path: Path
    wavelength_m: np.ndarray
    quantum_efficiency: np.ndarray
    optical_throughput: np.ndarray


_REQUIRED_TOP_LEVEL = {
    "schema_version",
    "enabled",
    "snr_threshold",
    "optics",
    "detector",
    "spectral_response",
    "asteroid",
    "angular_rate",
    "environment",
    "stray_light",
    "options",
}


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _require_finite_positive(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}.")
    return number


def _require_probability(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}.")
    return number


def _resolve_spectral_path(
    file_path: str | Path,
    *,
    config_path: str | Path | None,
    base_dir: str | Path | None,
) -> Path:
    path = Path(file_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base_dir is not None:
        return (Path(base_dir).expanduser().resolve() / path).resolve()
    if config_path is not None:
        return (Path(config_path).expanduser().resolve().parent / path).resolve()
    return (Path.cwd() / path).resolve()


def validate_payload_mapping(payload_snr: Mapping[str, Any]) -> None:
    """Validate scalar and structural payload settings before loading NPZ data."""

    missing = sorted(_REQUIRED_TOP_LEVEL - set(payload_snr))
    if missing:
        raise KeyError(f"payload_snr is missing required keys: {missing}")

    if int(payload_snr["schema_version"]) != 1:
        raise ValueError("Only payload_snr.schema_version=1 is currently supported.")

    if not isinstance(payload_snr["enabled"], bool):
        raise TypeError("payload_snr.enabled must be boolean.")
    _require_finite_positive(payload_snr["snr_threshold"], "payload_snr.snr_threshold")

    optics = _require_mapping(payload_snr["optics"], "payload_snr.optics")
    for key in (
        "aperture_diameter_m",
        "focal_length_m",
        "pixel_scale_arcsec_per_px",
        "pixel_pitch_m",
        "psf_sigma_px",
    ):
        _require_finite_positive(optics[key], f"payload_snr.optics.{key}")

    detector = _require_mapping(payload_snr["detector"], "payload_snr.detector")
    _require_finite_positive(detector["exposure_time_s"], "payload_snr.detector.exposure_time_s")
    for key in (
        "dark_current_e_per_s_px",
        "read_noise_e_rms_per_px",
    ):
        number = float(detector[key])
        if not np.isfinite(number) or number < 0.0:
            raise ValueError(f"payload_snr.detector.{key} must be finite and nonnegative.")
    if not np.isfinite(float(detector["background_surface_brightness_mag_arcsec2"])):
        raise ValueError("payload_snr.detector.background_surface_brightness_mag_arcsec2 must be finite.")

    asteroid = _require_mapping(payload_snr["asteroid"], "payload_snr.asteroid")
    if not str(asteroid["absolute_magnitude_metadata_column"]).strip():
        raise ValueError("absolute_magnitude_metadata_column cannot be empty.")
    _require_probability(asteroid["geometric_albedo"], "payload_snr.asteroid.geometric_albedo")
    g12 = float(asteroid["g12"])
    if not np.isfinite(g12) or not 0.0 <= g12 <= 1.0:
        raise ValueError("payload_snr.asteroid.g12 must lie in [0, 1].")

    angular_rate = _require_mapping(payload_snr["angular_rate"], "payload_snr.angular_rate")
    if angular_rate.get("mode") != "state_derived":
        raise ValueError("Stage-2 production configuration requires angular_rate.mode='state_derived'.")

    environment = _require_mapping(payload_snr["environment"], "payload_snr.environment")
    _require_finite_positive(
        environment["astronomical_unit_km"],
        "payload_snr.environment.astronomical_unit_km",
    )
    _require_finite_positive(environment["solar_radius_km"], "payload_snr.environment.solar_radius_km")
    _require_finite_positive(environment["solar_temperature_k"], "payload_snr.environment.solar_temperature_k")
    for body_name in ("earth", "moon"):
        body = _require_mapping(environment[body_name], f"payload_snr.environment.{body_name}")
        _require_finite_positive(body["radius_km"], f"payload_snr.environment.{body_name}.radius_km")
        _require_probability(body["geometric_albedo"], f"payload_snr.environment.{body_name}.geometric_albedo")
        if body["phase_law"] not in {"lambert", "lommel_seeliger"}:
            raise ValueError(f"Unsupported {body_name} phase_law: {body['phase_law']!r}.")

    stray = _require_mapping(payload_snr["stray_light"], "payload_snr.stray_light")
    angles = np.asarray(stray["off_axis_angle_deg"], dtype=float)
    efficiencies = np.asarray(stray["efficiency"], dtype=float)
    if angles.ndim != 1 or efficiencies.ndim != 1 or angles.size != efficiencies.size:
        raise ValueError("Stray-light angle and efficiency arrays must be equal-length 1D arrays.")
    if angles.size < 2 or np.any(~np.isfinite(angles)) or np.any(np.diff(angles) <= 0.0):
        raise ValueError("Stray-light angles must be finite and strictly increasing.")
    if np.any(~np.isfinite(efficiencies)) or np.any(efficiencies <= 0.0):
        raise ValueError("Stray-light efficiencies must be finite and positive.")
    if stray["interpolation"] != "log10":
        raise ValueError("Only log10 stray-light interpolation is currently supported.")
    if not isinstance(stray["hold_last_value"], bool):
        raise TypeError("stray_light.hold_last_value must be boolean.")
    zero_at_angle_deg = stray.get("zero_at_angle_deg")
    if zero_at_angle_deg is not None:
        zero_at_angle_deg = float(zero_at_angle_deg)
        if not np.isfinite(zero_at_angle_deg) or zero_at_angle_deg <= angles[-1]:
            raise ValueError(
                "stray_light.zero_at_angle_deg must be finite and greater "
                "than the final tabulated off-axis angle."
            )

    options = _require_mapping(payload_snr["options"], "payload_snr.options")
    for key in (
        "include_earth_double_reflection",
        "include_moon_double_reflection",
        "include_earth_stray_light",
        "include_moon_stray_light",
        "validate_inputs",
    ):
        if not isinstance(options[key], bool):
            raise TypeError(f"payload_snr.options.{key} must be boolean.")
    if options["aperture_pixel_mode"] not in {"continuous", "ceil"}:
        raise ValueError("aperture_pixel_mode must be 'continuous' or 'ceil'.")


def load_payload_configuration(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> LoadedPayloadConfiguration:
    """Load and validate ``payload_snr`` and its NPZ spectral response."""

    normalized = normalize_simulation_config(
        config,
        config_path=config_path,
    )
    if "payload_snr" not in normalized:
        raise KeyError("Configuration has no 'payload_snr' section.")
    payload_snr = dict(
        _require_mapping(normalized["payload_snr"], "payload_snr")
    )
    validate_payload_mapping(payload_snr)

    spectral = _require_mapping(payload_snr["spectral_response"], "payload_snr.spectral_response")
    spectral_path = _resolve_spectral_path(
        spectral["file_path"],
        config_path=config_path,
        base_dir=base_dir,
    )
    if not spectral_path.is_file():
        raise FileNotFoundError(f"Spectral response file not found: {spectral_path}")

    keys = {
        "wavelength": str(spectral["wavelength_key"]),
        "qe": str(spectral["quantum_efficiency_key"]),
        "throughput": str(spectral["optical_throughput_key"]),
    }
    with np.load(spectral_path, allow_pickle=False) as archive:
        missing = [key for key in keys.values() if key not in archive.files]
        if missing:
            raise KeyError(f"Spectral NPZ is missing arrays: {missing}")
        wavelength = np.asarray(archive[keys["wavelength"]], dtype=float).copy()
        qe = np.asarray(archive[keys["qe"]], dtype=float).copy()
        throughput = np.asarray(archive[keys["throughput"]], dtype=float).copy()

    if wavelength.ndim != 1 or qe.ndim != 1 or throughput.ndim != 1:
        raise ValueError("Spectral response arrays must be one-dimensional.")
    if not (wavelength.size == qe.size == throughput.size):
        raise ValueError(
            "Spectral response arrays must have identical lengths: "
            f"wavelength={wavelength.size}, QE={qe.size}, throughput={throughput.size}."
        )
    if wavelength.size < 2:
        raise ValueError("At least two spectral samples are required.")
    if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0) or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("Wavelengths must be finite, positive, and strictly increasing.")
    if np.any(~np.isfinite(qe)) or np.any((qe < 0.0) | (qe > 1.0)):
        raise ValueError("Quantum efficiency must be finite and lie in [0, 1].")
    if np.any(~np.isfinite(throughput)) or np.any((throughput < 0.0) | (throughput > 1.0)):
        raise ValueError("Optical throughput must be finite and lie in [0, 1].")

    return LoadedPayloadConfiguration(
        payload_snr=payload_snr,
        spectral_file_path=spectral_path,
        wavelength_m=wavelength,
        quantum_efficiency=qe,
        optical_throughput=throughput,
    )


def load_payload_configuration_from_yaml(
    yaml_path: str | Path,
) -> LoadedPayloadConfiguration:
    """Read a YAML file and return its validated payload configuration."""

    path = Path(yaml_path).expanduser().resolve()
    config = load_simulation_config(path)
    return load_payload_configuration(config, config_path=path)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a payload_snr YAML section and spectral NPZ.")
    parser.add_argument("config", help="Path to the overall YAML configuration.")
    args = parser.parse_args()

    loaded = load_payload_configuration_from_yaml(args.config)
    cfg = loaded.payload_snr
    print("Payload SNR configuration is valid.")
    print(f"  enabled: {cfg['enabled']}")
    print(f"  SNR threshold: {cfg['snr_threshold']}")
    print(f"  angular-rate mode: {cfg['angular_rate']['mode']}")
    print(f"  spectral file: {loaded.spectral_file_path}")
    print(f"  spectral samples: {loaded.wavelength_m.size}")
    print(
        "  wavelength range: "
        f"{loaded.wavelength_m[0] * 1e9:.2f}–{loaded.wavelength_m[-1] * 1e9:.2f} nm"
    )


if __name__ == "__main__":
    _main()
