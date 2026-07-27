from __future__ import annotations

"""
Single-epoch Earth--Moon invisibility-zone geometry.

The invisibility zone is represented by one circular cone:

* Its half-angle is a fixed, a-priori mission design constant defined in this
  module. It is not recomputed from the instantaneous Earth--Moon separation.
* Its time-varying axis is the normalized angular bisector of the
  spacecraft-to-Earth and spacecraft-to-Moon line-of-sight unit vectors.

SPICE is used only to query the Moon geocentric EME/J2000 position and the
Earth heliocentric ECLIPJ2000 position at the supplied JD TDB epoch. All
geocentric EME/ECLIP/SECR conversions are delegated to ``utilities.py``,
imported as ``util``.

Required kernels
----------------
A planetary SPK covering the requested epoch must be loaded. Call
``load_spice_meta_kernel`` once, pass a meta-kernel to
``compute_earth_moon_invisibility_zone``, or load kernels elsewhere before the
function call.

The SPICE positions are geometric (``ABCORR='NONE'``). This keeps the Earth,
Moon, spacecraft, and rotating-frame construction at one common coordinate
epoch. The half-angle should already include whatever operational margin is
required by the payload design.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

import utilities as util


FloatArray: TypeAlias = NDArray[np.float64]


# =============================================================================
# Mission configuration
# =============================================================================

# Fixed a-priori Earth--Moon invisibility-zone half-angle.
# Replace this value with the adopted mission requirement when finalized.
IV_ZONE_HALF_ANGLE_DEG = 12.0
IV_ZONE_HALF_ANGLE_RAD = float(np.deg2rad(IV_ZONE_HALF_ANGLE_DEG))

SPICE_MOON_TARGET = "MOON"
SPICE_EARTH_TARGET = "EARTH"
SPICE_SUN_OBSERVER = "SUN"
SPICE_EARTH_OBSERVER = "EARTH"
SPICE_EME_FRAME = "J2000"
SPICE_ECLIPTIC_FRAME = "ECLIPJ2000"
SPICE_ABERRATION_CORRECTION = "NONE"


# =============================================================================
# Result object
# =============================================================================


@dataclass(frozen=True)
class EarthMoonInvisibilityZone:
    """Single-epoch Earth--Moon invisibility-zone result.

    Parameters are returned in both geocentric EME/J2000 and geocentric SECR
    when useful for diagnostics. ``axis_secr`` and ``half_angle_rad`` are the
    two quantities required by the search-boresight optimizer.
    """

    jd_tdb: float
    et_tdb_seconds_past_j2000: float

    axis_secr: FloatArray
    axis_geo_eme: FloatArray
    half_angle_rad: float

    earth_los_secr: FloatArray
    moon_los_secr: FloatArray
    earth_los_geo_eme: FloatArray
    moon_los_geo_eme: FloatArray
    earth_moon_angular_separation_rad: float

    spacecraft_position_geo_eme_km: FloatArray
    spacecraft_position_secr_km: FloatArray
    moon_position_geo_eme_km: FloatArray
    moon_position_secr_km: FloatArray

    # This is returned so every caller can construct the same SECR frame at
    # this epoch using utilities.py without performing a second SPICE query.
    earth_heliocentric_eclip_position_km: FloatArray


# =============================================================================
# Validation and SPICE helpers
# =============================================================================


def _finite_vector3(value: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values.")
    return np.asarray(vector, dtype=float)


def _unit_vector(value: ArrayLike, name: str) -> FloatArray:
    vector = _finite_vector3(value, name)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return np.asarray(vector / norm, dtype=float)


def _require_spiceypy() -> Any:
    try:
        import spiceypy as spice
    except ImportError as exc:
        raise ImportError(
            "earth_moon_invisibility_zone.py requires spiceypy. Install "
            "spiceypy and furnish a planetary SPK/meta-kernel before use."
        ) from exc
    return spice


def load_spice_meta_kernel(meta_kernel_path: Path | str) -> None:
    """Load one SPICE meta-kernel using ``spiceypy.furnsh``."""

    path = Path(meta_kernel_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"SPICE meta-kernel not found: {path}")

    spice = _require_spiceypy()
    spice.furnsh(str(path))


def clear_spice_kernels() -> None:
    """Clear the process-wide SPICE kernel pool."""

    spice = _require_spiceypy()
    spice.kclear()


def jd_tdb_to_et(jd_tdb: float) -> float:
    """Convert Julian Date TDB to SPICE ET seconds past J2000."""

    jd = float(jd_tdb)
    if not np.isfinite(jd):
        raise ValueError("jd_tdb must be finite.")

    spice = _require_spiceypy()
    return float(spice.unitim(jd, "JDTDB", "ET"))


def query_single_epoch_spice_positions(
    jd_tdb: float,
) -> tuple[float, FloatArray, FloatArray]:
    """Query the positions needed to construct the SECR IV-zone geometry.

    Returns
    -------
    et_tdb_seconds_past_j2000
        SPICE ET corresponding to ``jd_tdb``.
    moon_position_geo_eme_km
        Moon position relative to Earth in J2000/EME axes.
    earth_heliocentric_eclip_position_km
        Earth position relative to the Sun in ECLIPJ2000 axes. This vector is
        required by the supplied SECR conversion utilities.
    """

    spice = _require_spiceypy()
    et = jd_tdb_to_et(jd_tdb)

    moon_geo_eme_km, _ = spice.spkpos(
        SPICE_MOON_TARGET,
        et,
        SPICE_EME_FRAME,
        SPICE_ABERRATION_CORRECTION,
        SPICE_EARTH_OBSERVER,
    )
    earth_helio_eclip_km, _ = spice.spkpos(
        SPICE_EARTH_TARGET,
        et,
        SPICE_ECLIPTIC_FRAME,
        SPICE_ABERRATION_CORRECTION,
        SPICE_SUN_OBSERVER,
    )

    return (
        et,
        _finite_vector3(moon_geo_eme_km, "moon_position_geo_eme_km"),
        _finite_vector3(
            earth_helio_eclip_km,
            "earth_heliocentric_eclip_position_km",
        ),
    )


# =============================================================================
# Frame conversion and IV-zone construction
# =============================================================================


def _geo_eme_position_to_secr(
    position_geo_eme_km: ArrayLike,
    earth_heliocentric_eclip_position_km: ArrayLike,
) -> FloatArray:
    """Convert one geocentric EME position to geocentric SECR using util."""

    position_eme = _finite_vector3(
        position_geo_eme_km,
        "position_geo_eme_km",
    )
    earth_helio_eclip = _finite_vector3(
        earth_heliocentric_eclip_position_km,
        "earth_heliocentric_eclip_position_km",
    )

    position_geo_eclip = util.geo_eme_to_geo_eclip_generic(
        position_eme,
        hint=("position",),
    )
    position_secr = util.geo_eclip_to_geo_secr_generic(
        position_geo_eclip,
        earth_helio_eclip,
        obj_hint=("position",),
        earth_hint=("position",),
    )
    return _finite_vector3(position_secr, "position_secr_km")


def compute_earth_moon_invisibility_zone(
    jd_tdb: float,
    spacecraft_position_geo_eme_km: ArrayLike,
    meta_kernel_path: Path | str | None = None,
) -> EarthMoonInvisibilityZone:
    """Compute the single-epoch Earth--Moon invisibility-zone cone.

    Parameters
    ----------
    jd_tdb
        Epoch as Julian Date TDB.
    spacecraft_position_geo_eme_km
        Spacecraft geocentric position in EME/J2000 axes, kilometres.
    meta_kernel_path
        Optional SPICE meta-kernel to furnish immediately before the query.
        Omit this when the required kernels are already loaded.

    Notes
    -----
    The returned half-angle is always ``IV_ZONE_HALF_ANGLE_RAD``. The
    instantaneous Earth--Moon angular separation is calculated only as a
    diagnostic and never changes the adopted half-angle.
    """

    if meta_kernel_path is not None:
        load_spice_meta_kernel(meta_kernel_path)

    spacecraft_geo_eme = _finite_vector3(
        spacecraft_position_geo_eme_km,
        "spacecraft_position_geo_eme_km",
    )

    et, moon_geo_eme, earth_helio_eclip = (
        query_single_epoch_spice_positions(jd_tdb)
    )

    spacecraft_secr = _geo_eme_position_to_secr(
        spacecraft_geo_eme,
        earth_helio_eclip,
    )
    moon_secr = _geo_eme_position_to_secr(
        moon_geo_eme,
        earth_helio_eclip,
    )

    earth_position_geo_eme = np.zeros(3, dtype=float)
    earth_position_secr = np.zeros(3, dtype=float)

    earth_los_geo_eme = _unit_vector(
        earth_position_geo_eme - spacecraft_geo_eme,
        "spacecraft_to_earth_geo_eme",
    )
    moon_los_geo_eme = _unit_vector(
        moon_geo_eme - spacecraft_geo_eme,
        "spacecraft_to_moon_geo_eme",
    )

    earth_los_secr = _unit_vector(
        earth_position_secr - spacecraft_secr,
        "spacecraft_to_earth_secr",
    )
    moon_los_secr = _unit_vector(
        moon_secr - spacecraft_secr,
        "spacecraft_to_moon_secr",
    )

    # The normalized sum is the great-circle angular bisector for two unit
    # directions, except for the degenerate antipodal case checked below.
    axis_geo_eme = _unit_vector(
        earth_los_geo_eme + moon_los_geo_eme,
        "earth_moon_bisector_geo_eme",
    )
    axis_secr = _unit_vector(
        earth_los_secr + moon_los_secr,
        "earth_moon_bisector_secr",
    )

    angular_separation = float(
        np.arccos(
            np.clip(
                float(earth_los_secr @ moon_los_secr),
                -1.0,
                1.0,
            )
        )
    )

    # The axes computed independently in EME and SECR should map to the same
    # physical direction. Verify this through the supplied utility functions.
    axis_geo_eme_as_secr = _geo_eme_position_to_secr(
        axis_geo_eme,
        earth_helio_eclip,
    )
    axis_geo_eme_as_secr = _unit_vector(
        axis_geo_eme_as_secr,
        "axis_geo_eme_as_secr",
    )
    if not np.allclose(axis_geo_eme_as_secr, axis_secr, atol=1.0e-11):
        raise RuntimeError(
            "EME- and SECR-derived IV-zone axes are inconsistent. Check "
            "the utilities.py frame conventions and supplied ephemerides."
        )

    return EarthMoonInvisibilityZone(
        jd_tdb=float(jd_tdb),
        et_tdb_seconds_past_j2000=et,
        axis_secr=axis_secr,
        axis_geo_eme=axis_geo_eme,
        half_angle_rad=IV_ZONE_HALF_ANGLE_RAD,
        earth_los_secr=earth_los_secr,
        moon_los_secr=moon_los_secr,
        earth_los_geo_eme=earth_los_geo_eme,
        moon_los_geo_eme=moon_los_geo_eme,
        earth_moon_angular_separation_rad=angular_separation,
        spacecraft_position_geo_eme_km=spacecraft_geo_eme,
        spacecraft_position_secr_km=spacecraft_secr,
        moon_position_geo_eme_km=moon_geo_eme,
        moon_position_secr_km=moon_secr,
        earth_heliocentric_eclip_position_km=earth_helio_eclip,
    )


if __name__ == "__main__":
    raise SystemExit(
        "Import this module and call compute_earth_moon_invisibility_zone(). "
        "A spacecraft geocentric EME position, JD TDB epoch, and loaded "
        "planetary SPICE kernel are required."
    )
