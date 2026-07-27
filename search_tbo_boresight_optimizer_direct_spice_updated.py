from __future__ import annotations

"""
Discrete search-phase boresight determination for a Sun--Earth--Moon L1 observer.

The algorithm maximizes detectable TBO residence time subject to a circular
payload FOV, a maximum boresight change, and a hard Earth--Moon invisibility
zone. Detectability is candidate-boresight dependent: every feasible candidate
is passed to the physical payload SNR model together with the residence cells
inside that candidate FOV. This allows the Earth/Moon stray-light terms to use
the actual commanded boresight rather than a point-to-grid surrogate.

Per observation window
----------------------
1. Load nonzero Cartesian residence cells in geocentric SECR coordinates.
2. Transform the previous EME/J2000 boresight into the current SECR frame.
3. Optionally rescore that current FOV with its actual boresight and retain it
   when its residence score remains above the configured fraction.
4. Generate candidate boresights inside the maximum-change cone.
5. Remove candidates whose complete FOV intersects the invisibility cone,
   including the configured angular margin.
6. For each feasible candidate, evaluate physical SNR only for cells inside its
   FOV, using that candidate as the payload boresight.
7. Sum residence time for cells satisfying the SNR threshold.
8. Select the maximum-score candidate and transform it back to EME/J2000 axes.

For production use, call ``build_single_epoch_search_geometry``. It invokes
the direct-SPICE Earth--Moon invisibility-zone module and returns a matched set
of observer, Sun, Earth, Moon, IV-zone, and SECR-frame data at one JD TDB epoch.

Frame convention
----------------
Residence cells and optimizer geometry use geocentric SECR. Payload SNR states
and returned inertial boresights use geocentric EME/J2000. The optimizer never
constructs its own EME/ECLIPJ2000/SECR rotation matrix. All such conversions are
delegated to ``utilities.py`` imported as ``util``.

Payload convention
------------------
``payload_reference_scenario.py`` contains the current reusable payload and
asteroid assumptions copied from the latest SNR contour-map script, including
H, geometric albedo, G12, and the representative apparent angular speed.

Dependencies
------------
Required: numpy, pandas, scipy, spiceypy, utilities.py,
          earth_moon_invisibility_zone_direct_spice.py,
          payload_asteroid_snr_model.py, payload_reference_scenario.py
SPICE files in the working directory: de430.bsp, naif0012.tls

Run directly in PyCharm. The minimum working example uses the SPICE-derived
Earth--Moon IV zone and the physical reusable payload scenario for both the
full search and previous-score shortcut paths.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, TypeAlias

import numpy as np
import pandas as pd
import spiceypy as sp
from numpy.typing import ArrayLike, NDArray

import utilities as util
import earth_moon_invisibility_zone_direct_spice as ivz
import payload_reference_scenario as reference_payload


FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
SNREvaluator: TypeAlias = Callable[[FloatArray, FloatArray], FloatArray]


# =============================================================================
# Status and result objects
# =============================================================================


class PointingStatus(str, Enum):
    """Explicit optimizer outcomes."""

    SUCCESS_CURRENT_BORESIGHT_RETAINED = (
        "success_current_boresight_retained"
    )
    SUCCESS_OPTIMIZED_BORESIGHT = "success_optimized_boresight"
    NO_RESIDENCE_CELLS = "no_residence_cells"
    NO_DETECTABLE_RESIDENCE_CELLS = "no_detectable_residence_cells"
    NO_FEASIBLE_CANDIDATE_BORESIGHTS = (
        "no_feasible_candidate_boresights"
    )
    ZERO_RESIDENCE_COVERAGE = "zero_residence_coverage"


@dataclass(frozen=True)
class PointingResult:
    """Result of one observation-window pointing calculation."""

    status: PointingStatus

    boresight_synodic: FloatArray | None
    boresight_inertial: FloatArray | None

    residence_score_days: float
    retained_current_boresight: bool

    number_of_residence_cells: int
    # Candidate-dependent count for the selected/retained FOV.
    number_of_detectable_cells: int | None
    number_of_candidate_boresights: int | None
    number_of_feasible_boresights: int | None
    # Geometric cell count inside the selected/retained FOV.
    number_of_cells_in_selected_fov: int

    reference_boresight_synodic: FloatArray


@dataclass(frozen=True)
class ResidenceGrid:
    """Sparse nonzero residence grid in the synodic frame."""

    positions_synodic_km: FloatArray
    residence_time_days: FloatArray


# =============================================================================
# Geocentric SECR / EME frame conversion through utilities.py
# =============================================================================


@dataclass(frozen=True)
class FrameContext:
    """Single-epoch data required by the supplied SECR conversion utilities.

    ``earth_heliocentric_eclip_position_km`` is the Earth position relative to
    the Sun in ECLIPJ2000 axes at the current epoch. The optimizer does not
    construct rotation matrices internally; every geocentric EME/ECLIP/SECR
    conversion is delegated to ``utilities.py``, imported as ``util``.
    """

    earth_heliocentric_eclip_position_km: ArrayLike

    def __post_init__(self) -> None:
        earth = np.asarray(
            self.earth_heliocentric_eclip_position_km,
            dtype=float,
        )
        if earth.shape != (3,):
            raise ValueError(
                "earth_heliocentric_eclip_position_km must have shape (3,)."
            )
        if not np.all(np.isfinite(earth)):
            raise ValueError(
                "earth_heliocentric_eclip_position_km must be finite."
            )
        if np.linalg.norm(earth) <= 0.0:
            raise ValueError(
                "earth_heliocentric_eclip_position_km must be nonzero."
            )
        object.__setattr__(
            self,
            "earth_heliocentric_eclip_position_km",
            earth,
        )


def _position_hint(values: ArrayLike) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim == 1 and array.shape == (3,):
        return ("position",)
    if array.ndim == 2 and array.shape[1] == 3:
        return ("batch", "position")
    raise ValueError("Frame-conversion input must have shape (3,) or (N, 3).")


def geo_eme_to_geo_secr(
    values_geo_eme: ArrayLike,
    frame_context: FrameContext,
) -> FloatArray:
    """Convert geocentric EME/J2000 vectors to geocentric SECR using util."""

    values = np.asarray(values_geo_eme, dtype=float)
    hint = _position_hint(values)
    values_geo_eclip = util.geo_eme_to_geo_eclip_generic(
        values,
        hint=hint,
    )
    values_secr = util.geo_eclip_to_geo_secr_generic(
        values_geo_eclip,
        frame_context.earth_heliocentric_eclip_position_km,
        obj_hint=hint,
        earth_hint=("position",),
    )
    return np.asarray(values_secr, dtype=float)


def geo_secr_to_geo_eme(
    values_secr: ArrayLike,
    frame_context: FrameContext,
) -> FloatArray:
    """Convert geocentric SECR vectors to geocentric EME/J2000 using util."""

    values = np.asarray(values_secr, dtype=float)
    hint = _position_hint(values)
    values_geo_eclip = util.geo_secr_to_geo_eclip_generic(
        values,
        frame_context.earth_heliocentric_eclip_position_km,
        obj_hint=hint,
        earth_hint=("position",),
    )
    values_geo_eme = util.geo_eclip_to_geo_eme_generic(
        values_geo_eclip,
        hint=hint,
    )
    return np.asarray(values_geo_eme, dtype=float)


@dataclass(frozen=True)
class SingleEpochSearchGeometry:
    """Matched single-epoch observer, body, IV-zone, and frame geometry."""

    jd_tdb: float
    et_tdb_seconds_past_j2000: float

    observer_position_synodic_km: FloatArray
    observer_position_geo_eme_km: FloatArray
    observer_velocity_geo_eme_km_s: FloatArray

    sun_position_geo_eme_km: FloatArray
    earth_position_geo_eme_km: FloatArray
    moon_position_geo_eme_km: FloatArray

    invisibility_zone_axis_synodic: FloatArray
    invisibility_zone_half_angle_rad: float
    frame_context: FrameContext
    invisibility_zone: ivz.EarthMoonInvisibilityZone


def _finite_vector3(value: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values.")
    return np.asarray(vector, dtype=float)


def build_single_epoch_search_geometry(
    jd_tdb: float,
    observer_position_geo_eme_km: ArrayLike,
    observer_velocity_geo_eme_km_s: ArrayLike | None = None,
) -> SingleEpochSearchGeometry:
    """Build matched SPICE, payload, IV-zone, and SECR geometry for one epoch.

    The IV-zone module directly furnishes ``de430.bsp`` and ``naif0012.tls``.
    Sun and Moon positions are geocentric EME/J2000 vectors at the same SPICE
    ET. Earth is the geocentric origin. When observer velocity is omitted, a
    zero EME/J2000 velocity is used; production ephemeris workflows should pass
    the actual observer velocity.
    """

    observer_position_eme = _finite_vector3(
        observer_position_geo_eme_km,
        "observer_position_geo_eme_km",
    )
    observer_velocity_eme = (
        np.zeros(3, dtype=float)
        if observer_velocity_geo_eme_km_s is None
        else _finite_vector3(
            observer_velocity_geo_eme_km_s,
            "observer_velocity_geo_eme_km_s",
        )
    )

    zone = ivz.compute_earth_moon_invisibility_zone(
        jd_tdb=jd_tdb,
        spacecraft_position_geo_eme_km=observer_position_eme,
    )
    frame_context = FrameContext(
        earth_heliocentric_eclip_position_km=(
            zone.earth_heliocentric_eclip_position_km
        )
    )
    observer_secr = np.asarray(
        geo_eme_to_geo_secr(observer_position_eme, frame_context),
        dtype=float,
    )

    if not np.allclose(
        observer_secr,
        zone.spacecraft_position_secr_km,
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Optimizer and IV-zone observer SECR positions are inconsistent."
        )

    sun_position_eme, _ = sp.spkpos(
        "SUN",
        zone.et_tdb_seconds_past_j2000,
        "J2000",
        "NONE",
        "EARTH",
    )

    return SingleEpochSearchGeometry(
        jd_tdb=float(jd_tdb),
        et_tdb_seconds_past_j2000=float(
            zone.et_tdb_seconds_past_j2000
        ),
        observer_position_synodic_km=observer_secr,
        observer_position_geo_eme_km=observer_position_eme,
        observer_velocity_geo_eme_km_s=observer_velocity_eme,
        sun_position_geo_eme_km=_finite_vector3(
            sun_position_eme,
            "sun_position_geo_eme_km",
        ),
        earth_position_geo_eme_km=np.zeros(3, dtype=float),
        moon_position_geo_eme_km=np.asarray(
            zone.moon_position_geo_eme_km,
            dtype=float,
        ),
        invisibility_zone_axis_synodic=np.asarray(
            zone.axis_secr,
            dtype=float,
        ),
        invisibility_zone_half_angle_rad=float(zone.half_angle_rad),
        frame_context=frame_context,
        invisibility_zone=zone,
    )


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class PointingConfig:
    """Discrete candidate, constraint, scoring, and shortcut settings."""

    snr_threshold: float
    fov_half_angle_rad: float
    maximum_boresight_change_rad: float
    candidate_angular_spacing_rad: float

    invisibility_zone_margin_rad: float = 0.0

    enable_previous_score_shortcut: bool = False
    shortcut_minimum_previous_score_fraction: float = 0.95

    snr_batch_size: int = 20_000
    candidate_batch_size: int = 128
    los_batch_size: int = 50_000

    def __post_init__(self) -> None:
        finite_scalars = {
            "snr_threshold": self.snr_threshold,
            "fov_half_angle_rad": self.fov_half_angle_rad,
            "maximum_boresight_change_rad": (
                self.maximum_boresight_change_rad
            ),
            "candidate_angular_spacing_rad": (
                self.candidate_angular_spacing_rad
            ),
            "invisibility_zone_margin_rad": (
                self.invisibility_zone_margin_rad
            ),
            "shortcut_minimum_previous_score_fraction": (
                self.shortcut_minimum_previous_score_fraction
            ),
        }
        for name, value in finite_scalars.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        if self.snr_threshold < 0.0:
            raise ValueError("snr_threshold must be nonnegative.")
        if not 0.0 < self.fov_half_angle_rad < np.pi:
            raise ValueError("fov_half_angle_rad must lie in (0, pi).")
        if not 0.0 <= self.maximum_boresight_change_rad <= np.pi:
            raise ValueError(
                "maximum_boresight_change_rad must lie in [0, pi]."
            )
        if not 0.0 < self.candidate_angular_spacing_rad <= np.pi:
            raise ValueError(
                "candidate_angular_spacing_rad must lie in (0, pi]."
            )
        if self.invisibility_zone_margin_rad < 0.0:
            raise ValueError(
                "invisibility_zone_margin_rad must be nonnegative."
            )
        if not 0.0 <= self.shortcut_minimum_previous_score_fraction <= 1.0:
            raise ValueError(
                "shortcut_minimum_previous_score_fraction must lie in [0, 1]."
            )
        if self.snr_batch_size < 1:
            raise ValueError("snr_batch_size must be >= 1.")
        if self.candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be >= 1.")
        if self.los_batch_size < 1:
            raise ValueError("los_batch_size must be >= 1.")


# =============================================================================
# Residence-grid I/O
# =============================================================================


DEFAULT_X_COLUMN = "Synodic x (km)"
DEFAULT_Y_COLUMN = "Synodic y (km)"
DEFAULT_Z_COLUMN = "Synodic z (km)"
DEFAULT_RESIDENCE_COLUMN = "residence_time_days"


def load_sparse_residence_grid_csv(
    csv_path: Path | str,
    x_column: str = DEFAULT_X_COLUMN,
    y_column: str = DEFAULT_Y_COLUMN,
    z_column: str = DEFAULT_Z_COLUMN,
    residence_column: str = DEFAULT_RESIDENCE_COLUMN,
) -> ResidenceGrid:
    """Load and validate the sparse 3D residence CSV."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Residence grid not found: {path}")

    frame = pd.read_csv(path)
    required = [x_column, y_column, z_column, residence_column]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(
            f"Residence CSV is missing columns {missing}. "
            f"Found: {list(frame.columns)}"
        )

    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    numeric = numeric[numeric[residence_column] > 0.0]

    positions = numeric[[x_column, y_column, z_column]].to_numpy(
        dtype=float,
    )
    residence = numeric[residence_column].to_numpy(dtype=float)

    return ResidenceGrid(
        positions_synodic_km=positions,
        residence_time_days=residence,
    )


# =============================================================================
# Vector utilities
# =============================================================================


def _unit_vector(vector: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(vector, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return np.asarray(array / norm, dtype=float)


def _orthogonal_basis(reference_unit: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return two unit vectors perpendicular to a unit reference vector."""

    helper_axes = np.eye(3)
    helper = helper_axes[
        int(np.argmin(np.abs(helper_axes @ reference_unit)))
    ]
    first = np.cross(reference_unit, helper)
    first /= np.linalg.norm(first)
    second = np.cross(reference_unit, first)
    second /= np.linalg.norm(second)
    return np.asarray(first), np.asarray(second)


def observer_relative_los(
    positions_synodic_km: ArrayLike,
    observer_position_synodic_km: ArrayLike,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Return LOS unit vectors, ranges, and valid-positive-range mask."""

    positions = np.asarray(positions_synodic_km, dtype=float)
    observer = np.asarray(observer_position_synodic_km, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_synodic_km must have shape (N, 3).")
    if observer.shape != (3,):
        raise ValueError(
            "observer_position_synodic_km must have shape (3,)."
        )

    relative = positions - observer[None, :]
    ranges = np.linalg.norm(relative, axis=1)
    valid = np.isfinite(ranges) & (ranges > 0.0)

    los = np.empty_like(relative, dtype=float)
    los[:] = np.nan
    los[valid] = relative[valid] / ranges[valid, None]
    return los, np.asarray(ranges), np.asarray(valid)


# =============================================================================
# Candidate generation and hard invisibility-zone filtering
# =============================================================================


def generate_spherical_cap_candidates(
    reference_boresight_unit: ArrayLike,
    maximum_angle_rad: float,
    angular_spacing_rad: float,
) -> FloatArray:
    """Generate approximately uniform candidates in a spherical cap.

    The maximum boresight-change constraint is built into this generator.
    Ring populations scale with ``sin(delta)`` to avoid using a constant number
    of azimuth samples on every ring.
    """

    reference = _unit_vector(
        reference_boresight_unit,
        "reference_boresight_unit",
    )

    if not 0.0 <= maximum_angle_rad <= np.pi:
        raise ValueError("maximum_angle_rad must lie in [0, pi].")
    if not 0.0 < angular_spacing_rad <= np.pi:
        raise ValueError("angular_spacing_rad must lie in (0, pi].")

    if maximum_angle_rad == 0.0:
        return reference.reshape(1, 3)

    basis_1, basis_2 = _orthogonal_basis(reference)

    n_rings = max(1, int(np.ceil(maximum_angle_rad / angular_spacing_rad)))
    ring_angles = np.linspace(0.0, maximum_angle_rad, n_rings + 1)

    candidate_blocks: list[FloatArray] = [reference.reshape(1, 3)]

    for delta in ring_angles[1:]:
        circumference = 2.0 * np.pi * np.sin(delta)
        n_azimuth = max(
            1,
            int(np.ceil(circumference / angular_spacing_rad)),
        )

        azimuth = np.linspace(
            0.0,
            2.0 * np.pi,
            n_azimuth,
            endpoint=False,
        )
        transverse = (
            np.cos(azimuth)[:, None] * basis_1[None, :]
            + np.sin(azimuth)[:, None] * basis_2[None, :]
        )
        ring = (
            np.cos(delta) * reference[None, :]
            + np.sin(delta) * transverse
        )
        ring /= np.linalg.norm(ring, axis=1, keepdims=True)
        candidate_blocks.append(np.asarray(ring, dtype=float))

    candidates = np.vstack(candidate_blocks)

    # Remove only machine-level duplicates, such as the unique antipode at pi.
    rounded = np.round(candidates, decimals=14)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    candidates = candidates[np.sort(unique_indices)]

    # Numerical validation of the built-in maximum-angle condition.
    minimum_alignment = np.cos(maximum_angle_rad) - 1.0e-12
    if np.any(candidates @ reference < minimum_alignment):
        raise RuntimeError(
            "Candidate generator produced a direction outside the requested cap."
        )

    return np.asarray(candidates, dtype=float)


def invisibility_zone_feasible_mask(
    candidate_boresights_synodic: ArrayLike,
    invisibility_zone_axis_synodic: ArrayLike,
    invisibility_zone_half_angle_rad: float,
    fov_half_angle_rad: float,
    invisibility_zone_margin_rad: float = 0.0,
) -> BoolArray:
    """Test whether each complete circular FOV lies outside the IZ cone."""

    candidates = np.asarray(candidate_boresights_synodic, dtype=float)
    axis = _unit_vector(
        invisibility_zone_axis_synodic,
        "invisibility_zone_axis_synodic",
    )

    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError(
            "candidate_boresights_synodic must have shape (M, 3)."
        )

    minimum_axis_separation = (
        float(invisibility_zone_half_angle_rad)
        + float(fov_half_angle_rad)
        + float(invisibility_zone_margin_rad)
    )

    if minimum_axis_separation > np.pi:
        return np.zeros(candidates.shape[0], dtype=bool)

    return np.asarray(
        candidates @ axis
        <= np.cos(minimum_axis_separation) + 1.0e-15,
        dtype=bool,
    )


def single_boresight_is_iz_feasible(
    boresight_synodic: ArrayLike,
    invisibility_zone_axis_synodic: ArrayLike,
    invisibility_zone_half_angle_rad: float,
    config: PointingConfig,
) -> bool:
    mask = invisibility_zone_feasible_mask(
        np.asarray(boresight_synodic, dtype=float).reshape(1, 3),
        invisibility_zone_axis_synodic,
        invisibility_zone_half_angle_rad,
        config.fov_half_angle_rad,
        config.invisibility_zone_margin_rad,
    )
    return bool(mask[0])


# =============================================================================
# Candidate-dependent SNR and FOV scoring
# =============================================================================


def _normalize_boresight_batch(
    boresights_synodic: ArrayLike,
    number_of_rows: int,
) -> FloatArray:
    """Return one normalized boresight per SNR-evaluation row."""

    boresights = np.asarray(boresights_synodic, dtype=float)
    if boresights.shape == (3,):
        boresights = np.broadcast_to(
            _unit_vector(boresights, "boresight_synodic"),
            (number_of_rows, 3),
        ).copy()
    elif boresights.shape == (number_of_rows, 3):
        norms = np.linalg.norm(boresights, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError(
                "Every candidate boresight must be finite and nonzero."
            )
        boresights = boresights / norms[:, None]
    else:
        raise ValueError(
            "boresights_synodic must have shape (3,) or (N, 3) matching "
            "the position batch."
        )
    return np.asarray(boresights, dtype=float)


def evaluate_snr_in_batches(
    positions_synodic_km: FloatArray,
    boresights_synodic: ArrayLike,
    snr_evaluator: SNREvaluator,
    batch_size: int,
) -> FloatArray:
    """Evaluate candidate-aware SNR in position/boresight batches."""

    positions = np.asarray(positions_synodic_km, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_synodic_km must have shape (N, 3).")

    boresights = _normalize_boresight_batch(
        boresights_synodic,
        positions.shape[0],
    )
    output = np.empty(positions.shape[0], dtype=float)

    for start in range(0, positions.shape[0], batch_size):
        stop = min(start + batch_size, positions.shape[0])
        values = np.asarray(
            snr_evaluator(
                positions[start:stop],
                boresights[start:stop],
            ),
            dtype=float,
        ).reshape(-1)
        if values.size != stop - start:
            raise ValueError(
                "snr_evaluator must return one SNR value per input row."
            )
        output[start:stop] = values

    return output


def fov_membership_mask(
    boresight_unit: ArrayLike,
    los_unit_vectors: FloatArray,
    fov_half_angle_rad: float,
) -> BoolArray:
    """Return the cells whose LOS lies inside one circular FOV."""

    boresight = _unit_vector(boresight_unit, "boresight_unit")
    los = np.asarray(los_unit_vectors, dtype=float)
    if los.ndim != 2 or los.shape[1] != 3:
        raise ValueError("los_unit_vectors must have shape (N, 3).")
    return np.asarray(
        los @ boresight >= np.cos(fov_half_angle_rad),
        dtype=bool,
    )


def score_single_boresight_with_local_snr(
    boresight_synodic: FloatArray,
    positions_synodic_km: FloatArray,
    residence_days: FloatArray,
    los_unit_vectors: FloatArray,
    snr_evaluator: SNREvaluator,
    config: PointingConfig,
) -> tuple[float, int, int]:
    """Score one boresight with physical SNR evaluated only in its FOV."""

    in_fov = fov_membership_mask(
        boresight_synodic,
        los_unit_vectors,
        config.fov_half_angle_rad,
    )
    selected_indices = np.flatnonzero(in_fov)
    if selected_indices.size == 0:
        return 0.0, 0, 0

    snr = evaluate_snr_in_batches(
        positions_synodic_km[selected_indices],
        boresight_synodic,
        snr_evaluator,
        config.snr_batch_size,
    )
    detectable = np.isfinite(snr) & (snr >= config.snr_threshold)
    score = float(np.sum(residence_days[selected_indices][detectable]))
    return score, int(selected_indices.size), int(np.count_nonzero(detectable))


def score_candidates_with_candidate_dependent_snr(
    candidate_boresights: FloatArray,
    positions_synodic_km: FloatArray,
    los_unit_vectors: FloatArray,
    residence_days: FloatArray,
    snr_evaluator: SNREvaluator,
    config: PointingConfig,
) -> tuple[FloatArray, NDArray[np.int64], NDArray[np.int64]]:
    """Score candidates using each candidate as the physical payload boresight.

    Candidate and LOS blocks limit the geometric matrix size. For every block,
    only candidate/cell pairs inside the FOV are sent to the SNR model. Pairs
    are vectorized so one SNR call may contain several candidates, each with
    its own repeated commanded boresight.
    """

    candidates = np.asarray(candidate_boresights, dtype=float)
    positions = np.asarray(positions_synodic_km, dtype=float)
    los = np.asarray(los_unit_vectors, dtype=float)
    weights = np.asarray(residence_days, dtype=float)

    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_boresights must have shape (M, 3).")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_synodic_km must have shape (N, 3).")
    if los.shape != positions.shape:
        raise ValueError("los_unit_vectors must match positions shape.")
    if weights.shape != (positions.shape[0],):
        raise ValueError("residence_days must have shape (N,).")

    scores = np.zeros(candidates.shape[0], dtype=float)
    geometric_counts = np.zeros(candidates.shape[0], dtype=np.int64)
    detectable_counts = np.zeros(candidates.shape[0], dtype=np.int64)
    cosine_fov = float(np.cos(config.fov_half_angle_rad))
    for candidate_start in range(
        0,
        candidates.shape[0],
        config.candidate_batch_size,
    ):
        candidate_stop = min(
            candidate_start + config.candidate_batch_size,
            candidates.shape[0],
        )

        candidate_block = candidates[candidate_start:candidate_stop]
        block_size = candidate_block.shape[0]
        block_scores = np.zeros(block_size, dtype=float)
        block_geometric = np.zeros(block_size, dtype=np.int64)
        block_detectable = np.zeros(block_size, dtype=np.int64)
        print(config.los_batch_size)
        print(los.shape)
        for los_start in range(0, los.shape[0], config.los_batch_size):
            los_stop = min(los_start + config.los_batch_size, los.shape[0])
            los_block = los[los_start:los_stop]
            inside = candidate_block @ los_block.T >= cosine_fov
            candidate_rows, local_cell_columns = np.nonzero(inside)
            if candidate_rows.size == 0:
                continue

            cell_indices = los_start + local_cell_columns
            block_geometric += np.bincount(
                candidate_rows,
                minlength=block_size,
            ).astype(np.int64)

            pair_snr = evaluate_snr_in_batches(
                positions[cell_indices],
                candidate_block[candidate_rows],
                snr_evaluator,
                config.snr_batch_size,
            )
            detectable = (
                np.isfinite(pair_snr)
                & (pair_snr >= config.snr_threshold)
            )
            if not np.any(detectable):
                continue

            detectable_rows = candidate_rows[detectable]
            detectable_cells = cell_indices[detectable]
            block_detectable += np.bincount(
                detectable_rows,
                minlength=block_size,
            ).astype(np.int64)
            block_scores += np.bincount(
                detectable_rows,
                weights=weights[detectable_cells],
                minlength=block_size,
            )

        scores[candidate_start:candidate_stop] = block_scores
        geometric_counts[candidate_start:candidate_stop] = block_geometric
        detectable_counts[candidate_start:candidate_stop] = block_detectable

    return scores, geometric_counts, detectable_counts


# =============================================================================
# Production adapter to payload_asteroid_snr_model.py
# =============================================================================


@dataclass
class PayloadSNRGridEvaluator:
    """Candidate-aware adapter from SECR cells to the payload SNR model.

    Positions and commanded boresights are supplied in geocentric SECR. They
    are converted through ``utilities.py`` to geocentric EME/J2000 before the
    physical SNR model is evaluated. Each row may have a different candidate
    boresight, which is required for candidate-dependent Earth/Moon stray light.
    """

    frame_context: FrameContext
    payload: Any
    asteroid: Any

    observer_position_geo_eme_km: ArrayLike
    observer_velocity_geo_eme_km_s: ArrayLike
    asteroid_velocity_geo_eme_km_s: ArrayLike

    sun_position_geo_eme_km: ArrayLike
    earth_position_geo_eme_km: ArrayLike
    moon_position_geo_eme_km: ArrayLike

    environment: Any = None
    options: Any = None
    phase_model: Any = None
    asteroid_angular_rate_override_arcsec_s: float | None = None

    @classmethod
    def from_reference_scenario(
        cls,
        geometry: SingleEpochSearchGeometry,
        scenario: reference_payload.ReferencePayloadScenario | None = None,
    ) -> "PayloadSNRGridEvaluator":
        """Construct the evaluator from the reusable contour-map scenario."""

        selected = (
            reference_payload.make_reference_payload_scenario()
            if scenario is None
            else scenario
        )
        return cls(
            frame_context=geometry.frame_context,
            payload=selected.payload,
            asteroid=selected.asteroid,
            observer_position_geo_eme_km=(
                geometry.observer_position_geo_eme_km
            ),
            observer_velocity_geo_eme_km_s=(
                geometry.observer_velocity_geo_eme_km_s
            ),
            asteroid_velocity_geo_eme_km_s=(
                selected.asteroid_velocity_geo_eme_km_s
            ),
            sun_position_geo_eme_km=geometry.sun_position_geo_eme_km,
            earth_position_geo_eme_km=geometry.earth_position_geo_eme_km,
            moon_position_geo_eme_km=geometry.moon_position_geo_eme_km,
            environment=selected.environment,
            options=selected.options,
            asteroid_angular_rate_override_arcsec_s=(
                selected.asteroid_angular_rate_override_arcsec_s
            ),
        )

    def __post_init__(self) -> None:
        from payload_asteroid_snr_model import HG12PhaseModel

        vector_fields = {
            "observer_position_geo_eme_km": self.observer_position_geo_eme_km,
            "observer_velocity_geo_eme_km_s": (
                self.observer_velocity_geo_eme_km_s
            ),
            "sun_position_geo_eme_km": self.sun_position_geo_eme_km,
            "earth_position_geo_eme_km": self.earth_position_geo_eme_km,
            "moon_position_geo_eme_km": self.moon_position_geo_eme_km,
        }
        for name, value in vector_fields.items():
            setattr(self, name, _finite_vector3(value, name))

        asteroid_velocity = np.asarray(
            self.asteroid_velocity_geo_eme_km_s,
            dtype=float,
        )
        if asteroid_velocity.shape != (3,) and not (
            asteroid_velocity.ndim == 2
            and asteroid_velocity.shape[1] == 3
        ):
            raise ValueError(
                "asteroid_velocity_geo_eme_km_s must have shape (3,) or "
                "shape (N, 3)."
            )
        if not np.all(np.isfinite(asteroid_velocity)):
            raise ValueError(
                "asteroid_velocity_geo_eme_km_s must be finite."
            )
        self.asteroid_velocity_geo_eme_km_s = asteroid_velocity

        if self.asteroid_angular_rate_override_arcsec_s is not None:
            angular_rate = float(
                self.asteroid_angular_rate_override_arcsec_s
            )
            if not np.isfinite(angular_rate) or angular_rate < 0.0:
                raise ValueError(
                    "asteroid_angular_rate_override_arcsec_s must be "
                    "finite and nonnegative."
                )
            self.asteroid_angular_rate_override_arcsec_s = angular_rate

        if self.phase_model is None:
            self.phase_model = HG12PhaseModel.from_default_table()

    def __call__(
        self,
        positions_synodic_km: FloatArray,
        boresights_synodic: FloatArray,
    ) -> FloatArray:
        from payload_asteroid_snr_model import (
            EnvironmentConfig,
            ObservationGeometry,
            SNROptions,
            compute_asteroid_snr,
        )

        positions_synodic = np.asarray(positions_synodic_km, dtype=float)
        if positions_synodic.ndim != 2 or positions_synodic.shape[1] != 3:
            raise ValueError(
                "positions_synodic_km must have shape (N, 3)."
            )
        boresights_synodic_batch = _normalize_boresight_batch(
            boresights_synodic,
            positions_synodic.shape[0],
        )

        positions_eme = geo_secr_to_geo_eme(
            positions_synodic,
            self.frame_context,
        )
        boresights_eme = geo_secr_to_geo_eme(
            boresights_synodic_batch,
            self.frame_context,
        )
        boresight_norms = np.linalg.norm(boresights_eme, axis=1)
        if np.any(boresight_norms <= 0.0):
            raise ValueError("Converted EME boresights must be nonzero.")
        boresights_eme = boresights_eme / boresight_norms[:, None]

        relative_eme = (
            positions_eme
            - np.asarray(self.observer_position_geo_eme_km)[None, :]
        )
        ranges = np.linalg.norm(relative_eme, axis=1)
        if np.any(ranges <= 0.0):
            raise ValueError(
                "A residence cell coincides with the observer position."
            )

        asteroid_velocity = np.asarray(
            self.asteroid_velocity_geo_eme_km_s,
            dtype=float,
        )
        if asteroid_velocity.shape == (3,):
            asteroid_velocity = np.broadcast_to(
                asteroid_velocity,
                positions_eme.shape,
            )
        elif asteroid_velocity.shape != positions_eme.shape:
            raise ValueError(
                "A batched asteroid velocity must match the positions passed "
                "to this evaluator."
            )

        observation_geometry = ObservationGeometry(
            observer_position_km=np.asarray(
                self.observer_position_geo_eme_km
            ),
            observer_velocity_km_s=np.asarray(
                self.observer_velocity_geo_eme_km_s
            ),
            asteroid_position_km=positions_eme,
            asteroid_velocity_km_s=asteroid_velocity,
            sun_position_km=np.asarray(self.sun_position_geo_eme_km),
            earth_position_km=np.asarray(self.earth_position_geo_eme_km),
            moon_position_km=np.asarray(self.moon_position_geo_eme_km),
            boresight_unit_vector=boresights_eme,
            asteroid_angular_rate_arcsec_s=(
                self.asteroid_angular_rate_override_arcsec_s
            ),
        )

        result = compute_asteroid_snr(
            payload=self.payload,
            asteroid=self.asteroid,
            geometry=observation_geometry,
            environment=(
                self.environment
                if self.environment is not None
                else EnvironmentConfig()
            ),
            options=(
                self.options
                if self.options is not None
                else SNROptions()
            ),
            phase_model=self.phase_model,
        )
        return np.asarray(result.snr, dtype=float).reshape(-1)


# =============================================================================
# Main optimizer
# =============================================================================


def determine_search_boresight(
    residence_grid: ResidenceGrid,
    observer_position_synodic_km: ArrayLike,
    previous_boresight_eme: ArrayLike,
    previous_residence_score_days: float | None,
    invisibility_zone_axis_synodic: ArrayLike,
    invisibility_zone_half_angle_rad: float,
    frame_context: FrameContext,
    snr_evaluator: SNREvaluator,
    config: PointingConfig,
) -> PointingResult:
    """Determine one hard-constrained, candidate-dependent search boresight."""

    if not np.isfinite(invisibility_zone_half_angle_rad):
        raise ValueError("invisibility_zone_half_angle_rad must be finite.")
    if not 0.0 <= invisibility_zone_half_angle_rad < np.pi:
        raise ValueError(
            "invisibility_zone_half_angle_rad must lie in [0, pi)."
        )

    positions = np.asarray(
        residence_grid.positions_synodic_km,
        dtype=float,
    )
    residence_days = np.asarray(
        residence_grid.residence_time_days,
        dtype=float,
    )

    previous_boresight_eme_unit = _unit_vector(
        previous_boresight_eme,
        "previous_boresight_eme",
    )
    reference_synodic = _unit_vector(
        geo_eme_to_geo_secr(previous_boresight_eme_unit, frame_context),
        "reference_boresight_synodic",
    )
    iz_axis_synodic = _unit_vector(
        invisibility_zone_axis_synodic,
        "invisibility_zone_axis_synodic",
    )

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "residence_grid.positions_synodic_km must have shape (N, 3)."
        )
    if residence_days.shape != (positions.shape[0],):
        raise ValueError(
            "residence_grid.residence_time_days must have shape (N,)."
        )

    finite_positive = (
        np.all(np.isfinite(positions), axis=1)
        & np.isfinite(residence_days)
        & (residence_days > 1.0)
    )
    positions = positions[finite_positive]
    residence_days = residence_days[finite_positive]

    if positions.shape[0] == 0:
        return PointingResult(
            status=PointingStatus.NO_RESIDENCE_CELLS,
            boresight_synodic=None,
            boresight_inertial=None,
            residence_score_days=0.0,
            retained_current_boresight=False,
            number_of_residence_cells=0,
            number_of_detectable_cells=None,
            number_of_candidate_boresights=None,
            number_of_feasible_boresights=None,
            number_of_cells_in_selected_fov=0,
            reference_boresight_synodic=reference_synodic,
        )

    los_all, _, valid_range = observer_relative_los(
        positions,
        observer_position_synodic_km,
    )
    positions = positions[valid_range]
    residence_days = residence_days[valid_range]
    los_all = los_all[valid_range]

    if positions.shape[0] == 0:
        return PointingResult(
            status=PointingStatus.NO_RESIDENCE_CELLS,
            boresight_synodic=None,
            boresight_inertial=None,
            residence_score_days=0.0,
            retained_current_boresight=False,
            number_of_residence_cells=0,
            number_of_detectable_cells=None,
            number_of_candidate_boresights=None,
            number_of_feasible_boresights=None,
            number_of_cells_in_selected_fov=0,
            reference_boresight_synodic=reference_synodic,
        )

    number_of_residence_cells = int(positions.shape[0])

    shortcut_eligible = (
        config.enable_previous_score_shortcut
        and previous_residence_score_days is not None
        and np.isfinite(previous_residence_score_days)
        and previous_residence_score_days > 0.0
        and single_boresight_is_iz_feasible(
            reference_synodic,
            iz_axis_synodic,
            invisibility_zone_half_angle_rad,
            config,
        )
    )

    if shortcut_eligible:
        current_score, n_current_geometric, n_current_detectable = (
            score_single_boresight_with_local_snr(
                boresight_synodic=reference_synodic,
                positions_synodic_km=positions,
                residence_days=residence_days,
                los_unit_vectors=los_all,
                snr_evaluator=snr_evaluator,
                config=config,
            )
        )
        retention_threshold = (
            config.shortcut_minimum_previous_score_fraction
            * float(previous_residence_score_days)
        )

        if current_score >= retention_threshold:
            retained_inertial = _unit_vector(
                geo_secr_to_geo_eme(reference_synodic, frame_context),
                "retained_boresight_eme",
            )
            return PointingResult(
                status=PointingStatus.SUCCESS_CURRENT_BORESIGHT_RETAINED,
                boresight_synodic=reference_synodic,
                boresight_inertial=retained_inertial,
                residence_score_days=current_score,
                retained_current_boresight=True,
                number_of_residence_cells=number_of_residence_cells,
                number_of_detectable_cells=n_current_detectable,
                number_of_candidate_boresights=0,
                number_of_feasible_boresights=0,
                number_of_cells_in_selected_fov=n_current_geometric,
                reference_boresight_synodic=reference_synodic,
            )

    candidates = generate_spherical_cap_candidates(
        reference_boresight_unit=reference_synodic,
        maximum_angle_rad=config.maximum_boresight_change_rad,
        angular_spacing_rad=config.candidate_angular_spacing_rad,
    )
    n_candidates = int(candidates.shape[0])

    iz_feasible = invisibility_zone_feasible_mask(
        candidate_boresights_synodic=candidates,
        invisibility_zone_axis_synodic=iz_axis_synodic,
        invisibility_zone_half_angle_rad=invisibility_zone_half_angle_rad,
        fov_half_angle_rad=config.fov_half_angle_rad,
        invisibility_zone_margin_rad=(
            config.invisibility_zone_margin_rad
        ),
    )
    feasible_candidates = candidates[iz_feasible]
    n_feasible = int(feasible_candidates.shape[0])

    if n_feasible == 0:
        return PointingResult(
            status=PointingStatus.NO_FEASIBLE_CANDIDATE_BORESIGHTS,
            boresight_synodic=None,
            boresight_inertial=None,
            residence_score_days=0.0,
            retained_current_boresight=False,
            number_of_residence_cells=number_of_residence_cells,
            number_of_detectable_cells=None,
            number_of_candidate_boresights=n_candidates,
            number_of_feasible_boresights=0,
            number_of_cells_in_selected_fov=0,
            reference_boresight_synodic=reference_synodic,
        )

    scores, geometric_counts, detectable_counts = (
        score_candidates_with_candidate_dependent_snr(
            candidate_boresights=feasible_candidates,
            positions_synodic_km=positions,
            los_unit_vectors=los_all,
            residence_days=residence_days,
            snr_evaluator=snr_evaluator,
            config=config,
        )
    )

    if int(np.max(detectable_counts, initial=0)) == 0:
        return PointingResult(
            status=PointingStatus.NO_DETECTABLE_RESIDENCE_CELLS,
            boresight_synodic=None,
            boresight_inertial=None,
            residence_score_days=0.0,
            retained_current_boresight=False,
            number_of_residence_cells=number_of_residence_cells,
            number_of_detectable_cells=0,
            number_of_candidate_boresights=n_candidates,
            number_of_feasible_boresights=n_feasible,
            number_of_cells_in_selected_fov=0,
            reference_boresight_synodic=reference_synodic,
        )

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    if best_score <= 0.0:
        return PointingResult(
            status=PointingStatus.ZERO_RESIDENCE_COVERAGE,
            boresight_synodic=None,
            boresight_inertial=None,
            residence_score_days=0.0,
            retained_current_boresight=False,
            number_of_residence_cells=number_of_residence_cells,
            number_of_detectable_cells=int(detectable_counts[best_index]),
            number_of_candidate_boresights=n_candidates,
            number_of_feasible_boresights=n_feasible,
            number_of_cells_in_selected_fov=int(
                geometric_counts[best_index]
            ),
            reference_boresight_synodic=reference_synodic,
        )

    selected_synodic = _unit_vector(
        feasible_candidates[best_index],
        "selected_boresight_synodic",
    )
    selected_inertial = _unit_vector(
        geo_secr_to_geo_eme(selected_synodic, frame_context),
        "selected_boresight_eme",
    )

    return PointingResult(
        status=PointingStatus.SUCCESS_OPTIMIZED_BORESIGHT,
        boresight_synodic=selected_synodic,
        boresight_inertial=selected_inertial,
        residence_score_days=best_score,
        retained_current_boresight=False,
        number_of_residence_cells=number_of_residence_cells,
        number_of_detectable_cells=int(detectable_counts[best_index]),
        number_of_candidate_boresights=n_candidates,
        number_of_feasible_boresights=n_feasible,
        number_of_cells_in_selected_fov=int(geometric_counts[best_index]),
        reference_boresight_synodic=reference_synodic,
    )


# =============================================================================
# Minimum working example
# =============================================================================


def _direction_from_axis_angles(
    axis_unit: ArrayLike,
    off_axis_angle_rad: float,
    azimuth_rad: float,
) -> FloatArray:
    """Return a direction offset from an arbitrary reference axis.

    Azimuth is measured in the local plane perpendicular to ``axis_unit``.
    """

    axis = _unit_vector(axis_unit, "axis_unit")
    basis_1, basis_2 = _orthogonal_basis(axis)
    direction = (
        np.cos(off_axis_angle_rad) * axis
        + np.sin(off_axis_angle_rad)
        * (
            np.cos(azimuth_rad) * basis_1
            + np.sin(azimuth_rad) * basis_2
        )
    )
    return _unit_vector(direction, "offset_direction")


def _make_demo_residence_csv(
    path: Path,
    observer_position_synodic_km: ArrayLike,
    invisibility_zone_axis_synodic: ArrayLike,
) -> None:
    """Create two residence lobes around the actual SPICE IV-zone axis."""

    rng = np.random.default_rng(17)
    observer = np.asarray(observer_position_synodic_km, dtype=float)
    iz_axis = _unit_vector(
        invisibility_zone_axis_synodic,
        "invisibility_zone_axis_synodic",
    )

    if observer.shape != (3,):
        raise ValueError(
            "observer_position_synodic_km must have shape (3,)."
        )

    rows: list[dict[str, float]] = []
    lobe_definitions = [
        # Stronger lobe 28 degrees from the Earth--Moon IV-zone axis.
        (np.deg2rad(28.0), np.deg2rad(0.0), 90, 9.0),
        # Weaker lobe on the opposite local azimuth.
        (np.deg2rad(35.0), np.deg2rad(180.0), 70, 5.5),
    ]

    for central_angle, central_azimuth, count, mean_weight in lobe_definitions:
        for _ in range(count):
            angle = central_angle + rng.normal(0.0, np.deg2rad(2.0))
            azimuth = central_azimuth + rng.normal(0.0, np.deg2rad(4.0))
            direction = _direction_from_axis_angles(
                iz_axis,
                angle,
                azimuth,
            )
            distance = rng.uniform(0.08e6, 0.65e6)
            position = observer + distance * direction
            residence = max(0.05, rng.normal(mean_weight, 1.2))
            rows.append(
                {
                    DEFAULT_X_COLUMN: position[0],
                    DEFAULT_Y_COLUMN: position[1],
                    DEFAULT_Z_COLUMN: position[2],
                    DEFAULT_RESIDENCE_COLUMN: residence,
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def minimum_working_example() -> None:
    """Run the SPICE IV zone and physical reusable payload SNR workflow.

    The working directory must contain ``de430.bsp`` and ``naif0012.tls``.
    The synthetic residence grid is only a compact optimizer test; SNR itself
    is evaluated by ``payload_asteroid_snr_model.py`` using the current payload,
    H, albedo, G12, and apparent-speed values from
    ``payload_reference_scenario.py``.
    """

    jd_tdb = 2451545.0

    ivz.furnish_spice_kernels()
    et = ivz.jd_tdb_to_et(jd_tdb)
    sun_position_geo_eme_km, _ = sp.spkpos(
        "SUN",
        et,
        "J2000",
        "NONE",
        "EARTH",
    )
    sun_direction_geo_eme = _unit_vector(
        sun_position_geo_eme_km,
        "sun_direction_geo_eme",
    )
    observer_position_geo_eme_km = 1.50e6 * sun_direction_geo_eme

    # The example uses a zero observer velocity because the reusable scenario
    # supplies an explicit apparent-angular-speed override. Production use
    # should pass the observer ephemeris velocity at the same JD TDB epoch.
    geometry = build_single_epoch_search_geometry(
        jd_tdb=jd_tdb,
        observer_position_geo_eme_km=observer_position_geo_eme_km,
        observer_velocity_geo_eme_km_s=np.zeros(3, dtype=float),
    )

    observer_synodic_km = geometry.observer_position_synodic_km
    iz_axis_synodic = geometry.invisibility_zone_axis_synodic
    iz_half_angle_rad = geometry.invisibility_zone_half_angle_rad
    frame_context = geometry.frame_context

    initial_boresight_synodic = _direction_from_axis_angles(
        iz_axis_synodic,
        np.deg2rad(25.0),
        0.0,
    )
    initial_boresight_eme = _unit_vector(
        geo_secr_to_geo_eme(initial_boresight_synodic, frame_context),
        "initial_boresight_eme",
    )

    config = PointingConfig(
        snr_threshold=5.0,
        fov_half_angle_rad=np.deg2rad(6.0),
        maximum_boresight_change_rad=np.deg2rad(15.0),
        candidate_angular_spacing_rad=np.deg2rad(1.5),
        invisibility_zone_margin_rad=np.deg2rad(1.0),
        enable_previous_score_shortcut=True,
        shortcut_minimum_previous_score_fraction=0.95,
        snr_batch_size=2_000,
        candidate_batch_size=64,
        los_batch_size=2_000,
    )

    scenario = reference_payload.make_reference_payload_scenario()
    snr_evaluator = PayloadSNRGridEvaluator.from_reference_scenario(
        geometry=geometry,
        scenario=scenario,
    )


    residence_csv = Path( "tbo_residence_time_results/xyz_synthetic_residence_grid_sparse.csv")
    _make_demo_residence_csv(
        residence_csv,
        observer_position_synodic_km=observer_synodic_km,
        invisibility_zone_axis_synodic=iz_axis_synodic,
    )
    residence_grid = load_sparse_residence_grid_csv(residence_csv)

    first = determine_search_boresight(
        residence_grid=residence_grid,
        observer_position_synodic_km=observer_synodic_km,
        previous_boresight_eme=initial_boresight_eme,
        previous_residence_score_days=None,
        invisibility_zone_axis_synodic=iz_axis_synodic,
        invisibility_zone_half_angle_rad=iz_half_angle_rad,
        frame_context=frame_context,
        snr_evaluator=snr_evaluator,
        config=config,
    )

    assert first.status == PointingStatus.SUCCESS_OPTIMIZED_BORESIGHT
    assert first.boresight_synodic is not None
    assert first.boresight_inertial is not None
    assert first.residence_score_days > 0.0
    assert first.number_of_detectable_cells is not None
    assert first.number_of_detectable_cells > 0
    assert single_boresight_is_iz_feasible(
        first.boresight_synodic,
        iz_axis_synodic,
        iz_half_angle_rad,
        config,
    )

    second = determine_search_boresight(
        residence_grid=residence_grid,
        observer_position_synodic_km=observer_synodic_km,
        previous_boresight_eme=first.boresight_inertial,
        previous_residence_score_days=first.residence_score_days,
        invisibility_zone_axis_synodic=iz_axis_synodic,
        invisibility_zone_half_angle_rad=iz_half_angle_rad,
        frame_context=frame_context,
        snr_evaluator=snr_evaluator,
        config=config,
    )

    assert (
        second.status
        == PointingStatus.SUCCESS_CURRENT_BORESIGHT_RETAINED
    )
    assert second.retained_current_boresight
    assert second.number_of_candidate_boresights == 0
    assert second.boresight_inertial is not None
    assert np.allclose(
        second.boresight_inertial,
        first.boresight_inertial,
        atol=1.0e-12,
    )

    zone = geometry.invisibility_zone
    print("Minimum working example completed successfully.")
    print(f"JD TDB:        {jd_tdb:.6f}")
    print(
        "IV-zone axis (SECR): "
        f"{np.array2string(iz_axis_synodic, precision=6)}"
    )
    print(
        "IV-zone half-angle: "
        f"{np.rad2deg(iz_half_angle_rad):.3f} deg"
    )
    print(
        "Earth--Moon separation: "
        f"{np.rad2deg(zone.earth_moon_angular_separation_rad):.3f} deg"
    )
    print(
        "Reference asteroid: "
        f"H={reference_payload.ASTEROID_ABSOLUTE_MAGNITUDE:.1f}, "
        f"p_V={reference_payload.ASTEROID_GEOMETRIC_ALBEDO:.3f}, "
        f"G12={reference_payload.ASTEROID_G12:.3f}, "
        f"speed="
        f"{reference_payload.ASTEROID_APPARENT_SPEED_ARCSEC_PER_HOUR:.1f} "
        "arcsec/hr"
    )
    print(f"First status:   {first.status.value}")
    print(f"First score:    {first.residence_score_days:.3f} days")
    print(
        "Selected FOV cells: "
        f"{first.number_of_cells_in_selected_fov} geometric, "
        f"{first.number_of_detectable_cells} detectable"
    )
    print(
        "First boresight (SECR): "
        f"{np.array2string(first.boresight_synodic, precision=6)}"
    )
    print(f"Second status:  {second.status.value}")
    print(f"Second score:   {second.residence_score_days:.3f} days")


if __name__ == "__main__":
    minimum_working_example()
