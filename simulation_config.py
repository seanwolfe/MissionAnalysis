from __future__ import annotations

"""Load, derive, and validate the mission simulation configuration.

The main mission YAML contains design and algorithm settings.  Stable physical
and unit constants are kept in a separate ``constants.yaml`` file and merged at
startup.  Derived values are calculated once here so downstream modules do not
maintain independent copies of the same quantity.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import math

import yaml


_LOADED_SPICE_KERNELS: set[str] = set()


def _read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} YAML root must be a mapping: {path}")
    return dict(value)


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}.")
    return number


def _nonnegative(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}.")
    return number


def _close(a: float, b: float, *, rtol: float = 1.0e-8, atol: float = 1.0e-12) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def _resolve_relative_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def normalize_simulation_config(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a deep-copied, constants-merged, derived configuration.

    Backward-compatible aliases such as ``AU_TO_M``, ``sigma_ra``, and
    ``time_between_frames`` are populated in memory for mature modules.  They
    no longer need to be repeated in the mission YAML.
    """

    cfg = deepcopy(dict(config))
    if bool(cfg.get("__normalized_config__", False)):
        return cfg

    if config_path is None:
        config_path = cfg.get("__config_path__")
    if config_path is not None:
        main_path = Path(config_path).expanduser().resolve()
        base_dir = main_path.parent
        cfg["__config_path__"] = str(main_path)
        cfg["__config_dir__"] = str(base_dir)
    else:
        main_path = None
        base_dir = Path(cfg.get("__config_dir__", Path.cwd())).expanduser().resolve()
        cfg["__config_dir__"] = str(base_dir)

    constants_value = cfg.get("constants_file")
    if constants_value is None:
        raise KeyError(
            "The mission configuration must define constants_file, for example "
            "constants_file: 'constants.yaml'."
        )
    constants_path = _resolve_relative_path(constants_value, base_dir)
    constants = _read_yaml_mapping(constants_path, "Constants")
    cfg["constants_file"] = str(constants_value)
    cfg["__constants_path__"] = str(constants_path)
    cfg["constants"] = deepcopy(constants)

    required_constants = (
        "AU_TO_M",
        "KM_TO_M",
        "SECONDS_PER_DAY",
        "MAS_TO_DEGREE",
        "GRAVITATIONAL_CONSTANT",
        "SUN_MASS",
        "MERCURY_MASS",
        "VENUS_MASS",
        "EARTH_MASS",
        "MOON_MASS",
        "MARS_MASS",
        "JUPITER_MASS",
        "SATURN_MASS",
        "URANUS_MASS",
        "NEPTUNE_MASS",
        "SUN_RADIUS_KM",
        "SUN_TEMPERATURE_K",
        "EARTH_RADIUS_KM",
        "MOON_RADIUS_KM",
        "EARTH_HILL_RADIUS_AU",
    )
    missing = [key for key in required_constants if key not in constants]
    if missing:
        raise KeyError(f"constants.yaml is missing required entries: {missing}")

    # Stable unit and physical aliases used by the existing framework. During
    # migration, reject conflicting duplicates instead of silently choosing one.
    for key in required_constants:
        if key in cfg:
            try:
                agrees = _close(float(cfg[key]), float(constants[key]))
            except (TypeError, ValueError):
                agrees = cfg[key] == constants[key]
            if not agrees:
                raise ValueError(
                    f"Main YAML {key}={cfg[key]!r} conflicts with "
                    f"constants.yaml {key}={constants[key]!r}. Remove the "
                    "duplicate from the main YAML."
                )
        cfg[key] = constants[key]

    au_to_m = _positive(cfg["AU_TO_M"], "constants.AU_TO_M")
    km_to_m = _positive(cfg["KM_TO_M"], "constants.KM_TO_M")
    seconds_per_day = _positive(
        cfg["SECONDS_PER_DAY"], "constants.SECONDS_PER_DAY"
    )
    mas_per_degree = _positive(
        cfg["MAS_TO_DEGREE"], "constants.MAS_TO_DEGREE"
    )
    gravitational_constant = _positive(
        cfg["GRAVITATIONAL_CONSTANT"],
        "constants.GRAVITATIONAL_CONSTANT",
    )

    # Write validated numeric values back into the runtime configuration.
    # PyYAML 1.1 may parse values such as ``1.989e30`` as strings unless the
    # exponent contains an explicit sign. Downstream numerical code must never
    # receive those raw string values.
    cfg["AU_TO_M"] = au_to_m
    cfg["KM_TO_M"] = km_to_m
    cfg["SECONDS_PER_DAY"] = seconds_per_day
    cfg["MAS_TO_DEGREE"] = mas_per_degree
    cfg["GRAVITATIONAL_CONSTANT"] = gravitational_constant

    body_names = (
        "SUN",
        "MERCURY",
        "VENUS",
        "EARTH",
        "MARS",
        "JUPITER",
        "SATURN",
        "URANUS",
        "NEPTUNE",
        "MOON",
    )
    body_masses = {
        name: _positive(cfg[f"{name}_MASS"], f"constants.{name}_MASS")
        for name in body_names
    }
    for name, mass in body_masses.items():
        cfg[f"{name}_MASS"] = mass

    g_km3_kg_s2 = gravitational_constant / km_to_m**3
    body_mu_km3_s2 = {
        name: g_km3_kg_s2 * mass
        for name, mass in body_masses.items()
    }
    cfg["BODY_MU_KM3_S2"] = body_mu_km3_s2
    naif_id_by_name = {
        "SUN": 10,
        "MERCURY": 1,
        "VENUS": 2,
        "EARTH": 399,
        "MARS": 4,
        "JUPITER": 5,
        "SATURN": 6,
        "URANUS": 7,
        "NEPTUNE": 8,
        "MOON": 301,
    }
    cfg["BODY_MU_BY_NAIF_KM3_S2"] = {
        naif_id_by_name[name]: mu
        for name, mu in body_mu_km3_s2.items()
    }
    derived_earth_mu = body_mu_km3_s2["EARTH"]
    if "EARTH_MASS_PARAMETER" in cfg and not _close(
        float(cfg["EARTH_MASS_PARAMETER"]), derived_earth_mu, rtol=1.0e-8
    ):
        raise ValueError(
            "EARTH_MASS_PARAMETER conflicts with G * EARTH_MASS. Remove the "
            "independent value from the main YAML."
        )
    cfg["EARTH_MASS_PARAMETER"] = derived_earth_mu

    sun_earth_moon_mass = (
        body_masses["SUN"] + body_masses["EARTH"] + body_masses["MOON"]
    )
    cfg["SUN_EARTH_MOON_MU_KM3_S2"] = (
        g_km3_kg_s2 * sun_earth_moon_mass
    )
    derived_system_mass_parameter = (
        body_masses["EARTH"] + body_masses["MOON"]
    ) / sun_earth_moon_mass
    if "SYSTEM_MASS_PARAMETER" in cfg and not _close(
        float(cfg["SYSTEM_MASS_PARAMETER"]),
        derived_system_mass_parameter,
        rtol=1.0e-8,
    ):
        raise ValueError(
            "SYSTEM_MASS_PARAMETER conflicts with the configured Sun/Earth/"
            "Moon masses. Remove the independent value from the main YAML."
        )
    cfg["SYSTEM_MASS_PARAMETER"] = derived_system_mass_parameter

    derived_hill_radius_km = (
        _positive(
            cfg["EARTH_HILL_RADIUS_AU"],
            "constants.EARTH_HILL_RADIUS_AU",
        )
        * au_to_m
        / km_to_m
    )
    if "EARTH_HILL_RADIUS_KM" in cfg and not _close(
        float(cfg["EARTH_HILL_RADIUS_KM"]),
        derived_hill_radius_km,
        rtol=1.0e-8,
    ):
        raise ValueError(
            "EARTH_HILL_RADIUS_KM conflicts with EARTH_HILL_RADIUS_AU * AU. "
            "Remove the independent value from the main YAML."
        )
    cfg["EARTH_HILL_RADIUS_KM"] = derived_hill_radius_km

    # ------------------------------------------------------------------
    # Optical geometry: pixel scale and pitch are authoritative.
    # small-angle relation: pixel_scale_rad_per_px = pixel_pitch_m / f_m
    # ------------------------------------------------------------------
    payload_snr = cfg.get("payload_snr")
    if not isinstance(payload_snr, Mapping):
        raise TypeError("payload_snr must be a YAML mapping.")
    payload_snr = deepcopy(dict(payload_snr))
    cfg["payload_snr"] = payload_snr

    optics = payload_snr.get("optics")
    if not isinstance(optics, Mapping):
        raise TypeError("payload_snr.optics must be a YAML mapping.")
    optics = deepcopy(dict(optics))
    payload_snr["optics"] = optics

    pixel_scale_arcsec_px = _positive(
        optics["pixel_scale_arcsec_per_px"],
        "payload_snr.optics.pixel_scale_arcsec_per_px",
    )
    pixel_pitch_m = _positive(
        optics["pixel_pitch_m"],
        "payload_snr.optics.pixel_pitch_m",
    )
    arcsec_to_rad = math.pi / (180.0 * 3600.0)
    focal_length_m = pixel_pitch_m / (
        pixel_scale_arcsec_px * arcsec_to_rad
    )

    existing_focal = optics.get("focal_length_m")
    if existing_focal is not None:
        existing_focal = _positive(
            existing_focal, "payload_snr.optics.focal_length_m"
        )
        if not _close(existing_focal, focal_length_m, rtol=1.0e-3):
            raise ValueError(
                "payload_snr.optics.focal_length_m conflicts with the value "
                "derived from pixel pitch and pixel scale: "
                f"configured={existing_focal:.12g} m, "
                f"derived={focal_length_m:.12g} m."
            )
    optics["focal_length_m"] = focal_length_m
    cfg["pixel_scale"] = pixel_scale_arcsec_px

    # ------------------------------------------------------------------
    # Measurement noise: centroiding is configured in pixels; pointing
    # remains an independent angular uncertainty in mas.
    # ------------------------------------------------------------------
    measurement_noise = cfg.get("measurement_noise")
    if not isinstance(measurement_noise, Mapping):
        raise TypeError("measurement_noise must be a YAML mapping.")
    measurement_noise = deepcopy(dict(measurement_noise))
    cfg["measurement_noise"] = measurement_noise

    sigma_ra_px = _nonnegative(
        measurement_noise["centroid_sigma_ra_px"],
        "measurement_noise.centroid_sigma_ra_px",
    )
    sigma_dec_px = _nonnegative(
        measurement_noise["centroid_sigma_dec_px"],
        "measurement_noise.centroid_sigma_dec_px",
    )
    sigma_pointing_mas = _nonnegative(
        measurement_noise["pointing_sigma_mas"],
        "measurement_noise.pointing_sigma_mas",
    )
    sigma_ra_mas = sigma_ra_px * pixel_scale_arcsec_px * 1000.0
    sigma_dec_mas = sigma_dec_px * pixel_scale_arcsec_px * 1000.0

    for legacy_key, derived_value in (
        ("sigma_ra", sigma_ra_mas),
        ("sigma_dec", sigma_dec_mas),
        ("sigma_pointing", sigma_pointing_mas),
    ):
        if legacy_key in cfg and not _close(
            float(cfg[legacy_key]), float(derived_value), rtol=1.0e-8
        ):
            raise ValueError(
                f"Legacy {legacy_key}={cfg[legacy_key]!r} conflicts with "
                f"the authoritative measurement_noise value "
                f"({derived_value:.12g} mas)."
            )
        cfg[legacy_key] = float(derived_value)

    # ------------------------------------------------------------------
    # Tracklet timing: exposure and dead/readout time define cadence.
    # The elapsed collection time ends at the end of the last exposure.
    # ------------------------------------------------------------------
    detector = payload_snr.get("detector")
    if not isinstance(detector, Mapping):
        raise TypeError("payload_snr.detector must be a YAML mapping.")
    detector = deepcopy(dict(detector))
    payload_snr["detector"] = detector

    exposure_time_s = _positive(
        detector["exposure_time_s"],
        "payload_snr.detector.exposure_time_s",
    )
    readout_dead_time_s = _nonnegative(
        detector.get("readout_dead_time_s", 0.0),
        "payload_snr.detector.readout_dead_time_s",
    )
    detector["readout_dead_time_s"] = readout_dead_time_s
    frame_cadence_s = exposure_time_s + readout_dead_time_s

    if "time_between_frames" in cfg and not _close(
        float(cfg["time_between_frames"]), frame_cadence_s
    ):
        raise ValueError(
            "time_between_frames conflicts with exposure_time_s + "
            "readout_dead_time_s."
        )
    cfg["time_between_frames"] = frame_cadence_s

    number_of_frames = int(cfg["number_of_frames"])
    if number_of_frames <= 0:
        raise ValueError("number_of_frames must be positive.")
    cfg["tracklet_collection_time_sec"] = (
        (number_of_frames - 1) * frame_cadence_s + exposure_time_s
    )

    if "NUMBER_OF_OBSERVATIONS" in cfg and int(
        cfg["NUMBER_OF_OBSERVATIONS"]
    ) != number_of_frames:
        raise ValueError(
            "NUMBER_OF_OBSERVATIONS and number_of_frames describe the same "
            "tracklet and must be equal."
        )
    cfg["NUMBER_OF_OBSERVATIONS"] = number_of_frames

    # ------------------------------------------------------------------
    # One radius source for occultation and payload photometry/stray light.
    # ------------------------------------------------------------------
    environment = payload_snr.get("environment")
    if not isinstance(environment, Mapping):
        raise TypeError("payload_snr.environment must be a YAML mapping.")
    environment = deepcopy(dict(environment))
    payload_snr["environment"] = environment
    environment["astronomical_unit_km"] = au_to_m / km_to_m
    environment["solar_radius_km"] = _positive(
        cfg["SUN_RADIUS_KM"], "constants.SUN_RADIUS_KM"
    )
    environment["solar_temperature_k"] = _positive(
        cfg["SUN_TEMPERATURE_K"], "constants.SUN_TEMPERATURE_K"
    )
    for body_name, radius_key in (
        ("earth", "EARTH_RADIUS_KM"),
        ("moon", "MOON_RADIUS_KM"),
    ):
        body = environment.get(body_name)
        if not isinstance(body, Mapping):
            raise TypeError(
                f"payload_snr.environment.{body_name} must be a YAML mapping."
            )
        body = deepcopy(dict(body))
        body["radius_km"] = _positive(
            cfg[radius_key], f"constants.{radius_key}"
        )
        environment[body_name] = body

    # ------------------------------------------------------------------
    # One full-FOV-to-IV-edge safety margin for initial packing and AC.
    # ------------------------------------------------------------------
    ems = cfg.get("ems")
    if not isinstance(ems, Mapping):
        raise TypeError("ems must be a YAML mapping.")
    ems = deepcopy(dict(ems))
    cfg["ems"] = ems

    if "fov_clearance_margin_deg" not in ems:
        raise KeyError(
            "ems.fov_clearance_margin_deg is required and replaces both "
            "ems.alpha_s_deg and "
            "initial_boresight_packing.iv_clearance_margin_deg."
        )
    clearance_deg = _nonnegative(
        ems["fov_clearance_margin_deg"],
        "ems.fov_clearance_margin_deg",
    )

    packing = cfg.get("initial_boresight_packing")
    if not isinstance(packing, Mapping):
        raise TypeError("initial_boresight_packing must be a YAML mapping.")
    packing = deepcopy(dict(packing))
    cfg["initial_boresight_packing"] = packing

    for legacy_name, legacy_value in (
        ("ems.alpha_s_deg", ems.get("alpha_s_deg")),
        (
            "initial_boresight_packing.iv_clearance_margin_deg",
            packing.get("iv_clearance_margin_deg"),
        ),
    ):
        if legacy_value is not None and not _close(
            float(legacy_value), clearance_deg
        ):
            raise ValueError(
                f"{legacy_name} conflicts with "
                "ems.fov_clearance_margin_deg."
            )
    ems["alpha_s_deg"] = clearance_deg
    packing["iv_clearance_margin_deg"] = clearance_deg
    ems.pop("p_em", None)
    ems.pop("R_em", None)

    dynamic_distance = _positive(
        ems.get("dynamic_virtual_distance_km", 1.0e6),
        "ems.dynamic_virtual_distance_km",
    )
    ems["dynamic_virtual_distance_km"] = dynamic_distance

    # ------------------------------------------------------------------
    # Explicit propagation settings; no hidden solver/body defaults.
    # ------------------------------------------------------------------
    propagator = cfg.get("n_body_propagator")
    if not isinstance(propagator, Mapping):
        raise TypeError("n_body_propagator must be a YAML mapping.")
    propagator = deepcopy(dict(propagator))
    cfg["n_body_propagator"] = propagator
    if not isinstance(propagator.get("bodies"), list) or not propagator["bodies"]:
        raise ValueError("n_body_propagator.bodies must be a non-empty list.")
    propagator["bodies"] = [str(body) for body in propagator["bodies"]]
    for key in ("frame", "origin", "method"):
        if not str(propagator.get(key, "")).strip():
            raise ValueError(f"n_body_propagator.{key} cannot be empty.")
    propagator["eps"] = _positive(
        propagator["eps"], "n_body_propagator.eps"
    )
    propagator["rtol"] = _positive(
        propagator["rtol"], "n_body_propagator.rtol"
    )
    propagator["atol"] = _positive(
        propagator["atol"], "n_body_propagator.atol"
    )

    legacy_n_body = cfg.get("legacy_n_body")
    if not isinstance(legacy_n_body, Mapping):
        raise TypeError("legacy_n_body must be a YAML mapping.")
    legacy_n_body = deepcopy(dict(legacy_n_body))
    cfg["legacy_n_body"] = legacy_n_body
    if not isinstance(legacy_n_body.get("bodies"), list) or not legacy_n_body["bodies"]:
        raise ValueError("legacy_n_body.bodies must be a non-empty list.")
    legacy_n_body["bodies"] = [str(body) for body in legacy_n_body["bodies"]]
    for key in ("frame", "reference_body", "earth_body", "method"):
        if not str(legacy_n_body.get(key, "")).strip():
            raise ValueError(f"legacy_n_body.{key} cannot be empty.")
    legacy_n_body["rtol"] = _positive(
        legacy_n_body["rtol"], "legacy_n_body.rtol"
    )
    legacy_n_body["atol"] = _positive(
        legacy_n_body["atol"], "legacy_n_body.atol"
    )
    close_approach = legacy_n_body.get("debug_close_approach_km")
    if close_approach is not None:
        legacy_n_body["debug_close_approach_km"] = _positive(
            close_approach, "legacy_n_body.debug_close_approach_km"
        )

    adcs = cfg.get("adcs")
    if not isinstance(adcs, Mapping):
        raise TypeError("adcs must be a YAML mapping.")
    adcs = deepcopy(dict(adcs))
    cfg["adcs"] = adcs
    adcs["slew_authority_factor"] = _positive(
        adcs["slew_authority_factor"],
        "adcs.slew_authority_factor",
    )

    spice_cfg = cfg.get("spice")
    if not isinstance(spice_cfg, Mapping):
        raise TypeError("spice must be a YAML mapping.")
    spice_cfg = deepcopy(dict(spice_cfg))
    cfg["spice"] = spice_cfg
    kernels = spice_cfg.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("spice.kernels must be a non-empty YAML list.")
    spice_cfg["kernels"] = [str(kernel) for kernel in kernels]

    # ------------------------------------------------------------------
    # Optional end-of-run mission outcome summary.
    # ------------------------------------------------------------------
    mission_summary = cfg.get("mission_summary", {}) or {}
    if not isinstance(mission_summary, Mapping):
        raise TypeError("mission_summary must be a YAML mapping.")
    mission_summary = deepcopy(dict(mission_summary))

    enabled = mission_summary.get("enabled", False)
    fail_on_error = mission_summary.get("fail_on_error", False)
    if not isinstance(enabled, bool):
        raise TypeError("mission_summary.enabled must be boolean.")
    if not isinstance(fail_on_error, bool):
        raise TypeError("mission_summary.fail_on_error must be boolean.")

    output_folder = str(
        mission_summary.get("output_folder", "mission_summary")
    ).strip()
    if not output_folder:
        raise ValueError("mission_summary.output_folder cannot be empty.")

    mission_summary["enabled"] = enabled
    mission_summary["output_folder"] = output_folder
    mission_summary["fail_on_error"] = fail_on_error
    cfg["mission_summary"] = mission_summary

    cfg["derived"] = {
        "focal_length_m": focal_length_m,
        "sigma_ra_mas": sigma_ra_mas,
        "sigma_dec_mas": sigma_dec_mas,
        "frame_cadence_s": frame_cadence_s,
        "tracklet_collection_time_sec": cfg[
            "tracklet_collection_time_sec"
        ],
        "system_mass_parameter": cfg["SYSTEM_MASS_PARAMETER"],
        "earth_mass_parameter_km3_s2": cfg["EARTH_MASS_PARAMETER"],
        "earth_hill_radius_km": cfg["EARTH_HILL_RADIUS_KM"],
    }
    cfg["__normalized_config__"] = True
    return cfg


def load_simulation_config(path: str | Path) -> dict[str, Any]:
    """Load the main mission YAML, merge constants, and derive shared values."""

    config_path = Path(path).expanduser().resolve()
    config = _read_yaml_mapping(config_path, "Mission configuration")
    return normalize_simulation_config(config, config_path=config_path)


def load_spice_kernels(spice_module: Any, config: Mapping[str, Any]) -> None:
    """Furnish the configured SPICE kernels once per Python process."""

    base_dir = Path(config.get("__config_dir__", Path.cwd())).resolve()
    spice_cfg = config["spice"]
    for kernel_value in spice_cfg["kernels"]:
        raw = Path(str(kernel_value)).expanduser()
        candidate = raw if raw.is_absolute() else (base_dir / raw)
        kernel_to_load = candidate if candidate.exists() else raw
        key = str(kernel_to_load)
        if key in _LOADED_SPICE_KERNELS:
            continue
        spice_module.furnsh(key)
        _LOADED_SPICE_KERNELS.add(key)
