from __future__ import annotations

"""
Single-epoch Earth--Moon invisibility-zone geometry.

The invisibility zone is represented by one circular cone:

* Its half-angle is a fixed, a-priori mission-design constant defined in this
  module. It is not recomputed from the instantaneous Earth--Moon separation.
* Its time-varying axis is the normalized angular bisector of the
  spacecraft-to-Earth and spacecraft-to-Moon line-of-sight unit vectors.

The epoch is supplied as Julian Date TDB. SPICE supplies the Moon geocentric
J2000 position and Earth heliocentric ECLIPJ2000 position. Every geocentric
EME/J2000, ECLIPJ2000, and SECR conversion is delegated to ``utilities.py``,
imported as ``util``.

The module assumes the following files are directly accessible from the
working directory (or otherwise resolvable by SPICE):

    de430.bsp
    naif0012.tls

They are furnished automatically with direct ``sp.furnsh`` calls.
"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import spiceypy as sp
from numpy.typing import ArrayLike, NDArray

import utilities as util


FloatArray: TypeAlias = NDArray[np.float64]


# =============================================================================
# Mission and SPICE configuration
# =============================================================================

# Fixed a-priori Earth--Moon invisibility-zone half-angle. This value is not
# changed using the instantaneous Earth--Moon angular separation.
IV_ZONE_HALF_ANGLE_DEG = 10.0
IV_ZONE_HALF_ANGLE_RAD = float(np.deg2rad(IV_ZONE_HALF_ANGLE_DEG))

PLANETARY_SPK = "de430.bsp"
LEAP_SECONDS_KERNEL = "naif0012.tls"

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

    ``axis_secr`` and ``half_angle_rad`` are the quantities required by the
    search-boresight optimizer. The remaining fields are returned for frame
    consistency and diagnostics.
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

    # Required by the provided SECR conversion utilities at this epoch.
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


def furnish_spice_kernels() -> None:
    """Furnish the fixed DE430 SPK and NAIF leap-seconds kernel."""

    sp.furnsh("de430.bsp")
    sp.furnsh("naif0012.tls")


def clear_spice_kernels() -> None:
    """Clear the process-wide SPICE kernel pool."""

    sp.kclear()


def jd_tdb_to_et(jd_tdb: float) -> float:
    """Convert Julian Date TDB to SPICE ET seconds past J2000."""

    jd = float(jd_tdb)
    if not np.isfinite(jd):
        raise ValueError("jd_tdb must be finite.")
    return float(sp.unitim(jd, "JDTDB", "ET"))


def query_single_epoch_spice_positions(
    jd_tdb: float,
) -> tuple[float, FloatArray, FloatArray]:
    """Query the Moon and Earth positions required by the IV-zone model.

    The required kernels are furnished directly inside this function.

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

    furnish_spice_kernels()
    et = jd_tdb_to_et(jd_tdb)

    moon_geo_eme_km, _ = sp.spkpos(
        SPICE_MOON_TARGET,
        et,
        SPICE_EME_FRAME,
        SPICE_ABERRATION_CORRECTION,
        SPICE_EARTH_OBSERVER,
    )
    earth_helio_eclip_km, _ = sp.spkpos(
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
) -> EarthMoonInvisibilityZone:
    """Compute the single-epoch Earth--Moon invisibility-zone cone.

    Parameters
    ----------
    jd_tdb
        Epoch as Julian Date TDB.
    spacecraft_position_geo_eme_km
        Spacecraft geocentric position in EME/J2000 axes, kilometres.

    Notes
    -----
    ``de430.bsp`` and ``naif0012.tls`` are furnished directly before the SPICE
    queries. The returned half-angle is always ``IV_ZONE_HALF_ANGLE_RAD``.
    The instantaneous Earth--Moon angular separation is diagnostic only.
    """

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

    # For two non-antipodal unit vectors, their normalized sum is the
    # great-circle angular bisector.
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

    # Verify that independently constructed EME and SECR axes represent the
    # same physical direction using the supplied utility conversions.
    axis_geo_eme_as_secr = _unit_vector(
        _geo_eme_position_to_secr(axis_geo_eme, earth_helio_eclip),
        "axis_geo_eme_as_secr",
    )
    if not np.allclose(
        axis_geo_eme_as_secr,
        axis_secr,
        atol=1.0e-11,
        rtol=0.0,
    ):
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
        "The working directory must contain de430.bsp and naif0012.tls."
    )
