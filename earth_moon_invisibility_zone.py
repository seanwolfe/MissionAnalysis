from __future__ import annotations

"""Dynamic Earth--Moon invisibility-zone geometry for detection filtering.

The invisibility zone is a circular cone with:

* a fixed mission-configured half-angle; and
* a time-varying axis equal to the normalized angular bisector of the
  spacecraft-to-Earth and spacecraft-to-Moon line-of-sight directions.

All Cartesian inputs to one call must share the same origin, axes, units, and
sample epochs. The geometric functions are therefore unit-agnostic. The SPICE
query helper returns Moon positions in geocentric J2000 kilometres.
"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

SPICE_MOON_TARGET = "MOON"
SPICE_EARTH_OBSERVER = "EARTH"
SPICE_EME_FRAME = "J2000"
SPICE_ABERRATION_CORRECTION = "NONE"


@dataclass(frozen=True)
class EarthMoonInvisibilityZoneBatch:
    """Dynamic IV-zone geometry for one epoch or a batch of epochs."""

    axis_geo_eme: FloatArray
    half_angle_rad: float
    earth_los_geo_eme: FloatArray
    moon_los_geo_eme: FloatArray
    earth_moon_angular_separation_rad: FloatArray


def _as_vector_batch(value: ArrayLike, name: str) -> tuple[FloatArray, bool]:
    """Return an ``(N, 3)`` array and whether the input was a single vector."""

    array = np.asarray(value, dtype=float)
    was_single = array.ndim == 1
    if was_single:
        if array.shape != (3,):
            raise ValueError(f"{name} must have shape (3,) or (N,3), got {array.shape}.")
        array = array.reshape(1, 3)
    elif array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (3,) or (N,3), got {array.shape}.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.asarray(array, dtype=float), was_single


def _broadcast_vector_batches(
    *named_values: tuple[str, ArrayLike],
) -> tuple[list[FloatArray], bool]:
    """Broadcast vector inputs to one common ``(N, 3)`` shape."""

    arrays: list[FloatArray] = []
    singles: list[bool] = []
    lengths: list[int] = []
    for name, value in named_values:
        array, was_single = _as_vector_batch(value, name)
        arrays.append(array)
        singles.append(was_single)
        lengths.append(array.shape[0])

    count = max(lengths)
    out: list[FloatArray] = []
    for (name, _), array in zip(named_values, arrays):
        if array.shape[0] == count:
            out.append(array)
        elif array.shape[0] == 1:
            out.append(np.broadcast_to(array, (count, 3)).copy())
        else:
            raise ValueError(
                f"{name} has {array.shape[0]} epochs but the common batch has {count}."
            )

    return out, all(singles)


def _normalize_rows(vectors: FloatArray, name: str) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=1)
    bad = norms <= 1.0e-15
    if np.any(bad):
        raise ValueError(
            f"{name} contains zero-length vectors at indices "
            f"{np.flatnonzero(bad).tolist()}."
        )
    return np.asarray(vectors / norms[:, None], dtype=float)


def validate_half_angle_deg(half_angle_deg: float) -> float:
    """Validate and convert the fixed IV-zone half-angle to radians."""

    value = float(half_angle_deg)
    if not np.isfinite(value) or not 0.0 <= value < 180.0:
        raise ValueError(
            f"EMS half_angle_deg must be finite and lie in [0, 180), got {half_angle_deg!r}."
        )
    return float(np.deg2rad(value))


def compute_earth_moon_invisibility_zone_batch(
    *,
    spacecraft_position: ArrayLike,
    earth_position: ArrayLike,
    moon_position: ArrayLike,
    half_angle_deg: float,
) -> EarthMoonInvisibilityZoneBatch:
    """Construct the dynamic Earth--Moon bisector cone.

    Positions may be expressed in any common Cartesian frame and distance unit.
    For the production detection path they are geocentric EME/J2000 positions.
    """

    (
        spacecraft,
        earth,
        moon,
    ), was_single = _broadcast_vector_batches(
        ("spacecraft_position", spacecraft_position),
        ("earth_position", earth_position),
        ("moon_position", moon_position),
    )

    earth_los = _normalize_rows(earth - spacecraft, "spacecraft-to-Earth LOS")
    moon_los = _normalize_rows(moon - spacecraft, "spacecraft-to-Moon LOS")

    bisector_sum = earth_los + moon_los
    bisector_norm = np.linalg.norm(bisector_sum, axis=1)
    antipodal = bisector_norm <= 1.0e-15
    if np.any(antipodal):
        raise ValueError(
            "Earth and Moon LOS directions are antipodal, so their angular "
            f"bisector is undefined at indices {np.flatnonzero(antipodal).tolist()}."
        )
    axis = bisector_sum / bisector_norm[:, None]

    separation = np.arccos(
        np.clip(np.einsum("ij,ij->i", earth_los, moon_los), -1.0, 1.0)
    )
    half_angle_rad = validate_half_angle_deg(half_angle_deg)

    if was_single:
        return EarthMoonInvisibilityZoneBatch(
            axis_geo_eme=np.asarray(axis[0], dtype=float),
            half_angle_rad=half_angle_rad,
            earth_los_geo_eme=np.asarray(earth_los[0], dtype=float),
            moon_los_geo_eme=np.asarray(moon_los[0], dtype=float),
            earth_moon_angular_separation_rad=np.asarray(separation[0], dtype=float),
        )

    return EarthMoonInvisibilityZoneBatch(
        axis_geo_eme=np.asarray(axis, dtype=float),
        half_angle_rad=half_angle_rad,
        earth_los_geo_eme=np.asarray(earth_los, dtype=float),
        moon_los_geo_eme=np.asarray(moon_los, dtype=float),
        earth_moon_angular_separation_rad=np.asarray(separation, dtype=float),
    )


def line_of_sight_inside_invisibility_zone(
    *,
    target_position: ArrayLike,
    spacecraft_position: ArrayLike,
    zone: EarthMoonInvisibilityZoneBatch,
) -> tuple[BoolArray | np.bool_, FloatArray]:
    """Test whether the spacecraft-to-target LOS lies inside the IV cone.

    Returns the Boolean inside-zone flag and the target-to-axis separation in
    radians. Boundary points are treated as inside the exclusion zone.
    """

    (target, spacecraft), was_single = _broadcast_vector_batches(
        ("target_position", target_position),
        ("spacecraft_position", spacecraft_position),
    )
    target_los = _normalize_rows(target - spacecraft, "spacecraft-to-target LOS")

    axis, axis_single = _as_vector_batch(zone.axis_geo_eme, "zone.axis_geo_eme")
    if axis.shape[0] == 1 and target.shape[0] > 1:
        axis = np.broadcast_to(axis, target.shape).copy()
    elif axis.shape[0] != target.shape[0]:
        raise ValueError(
            "zone.axis_geo_eme length does not match target/spacecraft batch: "
            f"axis={axis.shape[0]}, batch={target.shape[0]}."
        )
    axis = _normalize_rows(axis, "zone axis")

    separation = np.arccos(
        np.clip(np.einsum("ij,ij->i", target_los, axis), -1.0, 1.0)
    )
    inside = separation <= float(zone.half_angle_rad)

    if was_single and axis_single:
        return np.bool_(inside[0]), np.asarray(separation[0], dtype=float)
    return np.asarray(inside, dtype=bool), np.asarray(separation, dtype=float)


def query_moon_positions_geo_eme_km(jd_tdb: ArrayLike) -> FloatArray:
    """Query geocentric J2000 Moon positions at Julian Date TDB epochs.

    The caller must furnish an SPK containing the Earth and Moon and a leap-
    seconds kernel before calling this function. ``Spacecraft.py`` already
    furnishes ``de430.bsp`` and ``naif0012.tls`` once at import.
    """

    import spiceypy as sp

    epochs = np.asarray(jd_tdb, dtype=float)
    was_scalar = epochs.ndim == 0
    epochs = epochs.reshape(-1)
    if not np.all(np.isfinite(epochs)):
        raise ValueError("jd_tdb must contain only finite values.")

    moon_positions = np.empty((epochs.size, 3), dtype=float)
    for index, jd in enumerate(epochs):
        et = sp.unitim(float(jd), "JDTDB", "ET")
        position, _ = sp.spkpos(
            SPICE_MOON_TARGET,
            et,
            SPICE_EME_FRAME,
            SPICE_ABERRATION_CORRECTION,
            SPICE_EARTH_OBSERVER,
        )
        moon_positions[index] = np.asarray(position, dtype=float)

    if was_scalar:
        return np.asarray(moon_positions[0], dtype=float)
    return moon_positions


def compute_earth_moon_invisibility_zone(
    jd_tdb: float,
    spacecraft_position_geo_eme_km: ArrayLike,
    half_angle_deg: float,
) -> EarthMoonInvisibilityZoneBatch:
    """Single-epoch convenience wrapper using a direct SPICE Moon query."""

    spacecraft = np.asarray(spacecraft_position_geo_eme_km, dtype=float)
    moon = query_moon_positions_geo_eme_km(float(jd_tdb))
    return compute_earth_moon_invisibility_zone_batch(
        spacecraft_position=spacecraft,
        earth_position=np.zeros(3, dtype=float),
        moon_position=moon,
        half_angle_deg=half_angle_deg,
    )
