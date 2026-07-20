from __future__ import annotations

"""
Discrete search-phase boresight determination for a Sun--Earth--Moon L1 observer.

The algorithm maximizes only detectable TBO residence time. The SNR threshold
is deterministic and is applied as a binary eligibility mask to residence-grid
cells. The payload FOV and Earth--Moon invisibility zone are circular cones.

Per observation window
----------------------
1. Load nonzero Cartesian residence cells in the synodic frame.
2. Transform the previous inertial boresight into the current synodic frame.
3. Optionally evaluate only the current FOV and retain the current boresight if
   its score remains above a configured fraction of the previous score.
4. Otherwise evaluate SNR at every residence cell and remove cells below the
   threshold.
5. Generate candidate boresights inside the maximum-change cone about the
   current reference boresight.
6. Remove candidates whose complete FOV intersects the invisibility cone.
7. Sum detectable residence time for every cell whose observer-relative LOS
   lies inside each candidate FOV.
8. Select the maximum-score candidate and transform it back to inertial axes.

Frame convention
----------------
Residence cells are stored in an Earth-centred synodic frame. The optimizer
works in synodic directions, but the selected command is returned in both
synodic and inertial axes. The supplied 3x3 matrix R_I_from_S must satisfy

    v_I = R_I_from_S @ v_S

for direction vectors. Positions additionally include the supplied synodic
origin position in the inertial frame.

SNR convention
--------------
The optimizer accepts any callable that maps synodic Cartesian cell positions
with shape (N, 3) to deterministic SNR values with shape (N,). A helper class,
PayloadSNRGridEvaluator, connects this interface to
payload_asteroid_snr_model.py. It follows the existing point-to-grid SNR-map
convention: for each hypothetical cell, the model boresight is directed at that
cell. The candidate FOV test is applied separately by this optimizer.

Dependencies
------------
Required: numpy, pandas
Production SNR adapter: payload_asteroid_snr_model.py and its dependencies

Run directly in PyCharm. The minimum working example at the bottom creates a
small synthetic residence CSV, runs a full optimization, then verifies the
optional previous-score shortcut on a second call.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Protocol, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
SNREvaluator: TypeAlias = Callable[[FloatArray], FloatArray]


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
    number_of_detectable_cells: int | None
    number_of_candidate_boresights: int | None
    number_of_feasible_boresights: int | None
    number_of_cells_in_selected_fov: int

    reference_boresight_synodic: FloatArray


@dataclass(frozen=True)
class ResidenceGrid:
    """Sparse nonzero residence grid in the synodic frame."""

    positions_synodic_km: FloatArray
    residence_time_days: FloatArray


# =============================================================================
# Synodic/inertial transformation
# =============================================================================


@dataclass(frozen=True)
class SynodicInertialTransform:
    """Instantaneous rigid transformation from synodic to inertial axes.

    Parameters
    ----------
    rotation_inertial_from_synodic
        Proper orthonormal rotation matrix R_I_from_S satisfying
        ``direction_I = R_I_from_S @ direction_S``.
    synodic_origin_position_inertial_km
        Inertial position of the synodic-frame origin. For the residence CSV
        produced by the companion script, this will normally be Earth's
        inertial position because the grid is Earth-centred.
    """

    rotation_inertial_from_synodic: ArrayLike
    synodic_origin_position_inertial_km: ArrayLike

    def __post_init__(self) -> None:
        rotation = np.asarray(
            self.rotation_inertial_from_synodic,
            dtype=float,
        )
        origin = np.asarray(
            self.synodic_origin_position_inertial_km,
            dtype=float,
        )

        if rotation.shape != (3, 3):
            raise ValueError(
                "rotation_inertial_from_synodic must have shape (3, 3)."
            )
        if origin.shape != (3,):
            raise ValueError(
                "synodic_origin_position_inertial_km must have shape (3,)."
            )
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(origin)):
            raise ValueError("Frame transformation values must be finite.")

        identity_error = np.max(
            np.abs(rotation.T @ rotation - np.eye(3))
        )
        determinant = float(np.linalg.det(rotation))
        if identity_error > 1.0e-10 or not np.isclose(
            determinant,
            1.0,
            atol=1.0e-10,
        ):
            raise ValueError(
                "rotation_inertial_from_synodic must be a proper orthonormal "
                "rotation matrix."
            )

        object.__setattr__(
            self,
            "rotation_inertial_from_synodic",
            rotation,
        )
        object.__setattr__(
            self,
            "synodic_origin_position_inertial_km",
            origin,
        )

    @property
    def rotation_synodic_from_inertial(self) -> FloatArray:
        return np.asarray(self.rotation_inertial_from_synodic).T

    def directions_synodic_to_inertial(
        self,
        directions_synodic: ArrayLike,
    ) -> FloatArray:
        directions = np.asarray(directions_synodic, dtype=float)
        return np.asarray(
            directions @ np.asarray(
                self.rotation_inertial_from_synodic,
            ).T,
            dtype=float,
        )

    def directions_inertial_to_synodic(
        self,
        directions_inertial: ArrayLike,
    ) -> FloatArray:
        directions = np.asarray(directions_inertial, dtype=float)
        return np.asarray(
            directions @ self.rotation_synodic_from_inertial.T,
            dtype=float,
        )

    def positions_synodic_to_inertial(
        self,
        positions_synodic_km: ArrayLike,
    ) -> FloatArray:
        positions = np.asarray(positions_synodic_km, dtype=float)
        rotated = self.directions_synodic_to_inertial(positions)
        return np.asarray(
            rotated + np.asarray(
                self.synodic_origin_position_inertial_km,
            ),
            dtype=float,
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

    invisibility_zone_half_angle_rad: float
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
            "invisibility_zone_half_angle_rad": (
                self.invisibility_zone_half_angle_rad
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
        if not 0.0 <= self.invisibility_zone_half_angle_rad < np.pi:
            raise ValueError(
                "invisibility_zone_half_angle_rad must lie in [0, pi)."
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
    config: PointingConfig,
) -> bool:
    mask = invisibility_zone_feasible_mask(
        np.asarray(boresight_synodic, dtype=float).reshape(1, 3),
        invisibility_zone_axis_synodic,
        config.invisibility_zone_half_angle_rad,
        config.fov_half_angle_rad,
        config.invisibility_zone_margin_rad,
    )
    return bool(mask[0])


# =============================================================================
# SNR and FOV scoring
# =============================================================================


def evaluate_snr_in_batches(
    positions_synodic_km: FloatArray,
    snr_evaluator: SNREvaluator,
    batch_size: int,
) -> FloatArray:
    """Evaluate an arbitrary deterministic SNR callback in position batches."""

    positions = np.asarray(positions_synodic_km, dtype=float)
    output = np.empty(positions.shape[0], dtype=float)

    for start in range(0, positions.shape[0], batch_size):
        stop = min(start + batch_size, positions.shape[0])
        values = np.asarray(
            snr_evaluator(positions[start:stop]),
            dtype=float,
        ).reshape(-1)
        if values.size != stop - start:
            raise ValueError(
                "snr_evaluator must return one SNR value per input position."
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


def score_candidates_batched(
    candidate_boresights: FloatArray,
    detectable_los_unit_vectors: FloatArray,
    detectable_residence_days: FloatArray,
    fov_half_angle_rad: float,
    candidate_batch_size: int,
    los_batch_size: int,
) -> FloatArray:
    """Sum detectable residence time inside every candidate FOV.

    Cells remain separate. Therefore, Cartesian cells at different ranges but
    with identical or nearly identical LOS directions independently add their
    residence weights to every candidate that contains that LOS.
    """

    candidates = np.asarray(candidate_boresights, dtype=float)
    los = np.asarray(detectable_los_unit_vectors, dtype=float)
    weights = np.asarray(detectable_residence_days, dtype=float)

    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_boresights must have shape (M, 3).")
    if los.ndim != 2 or los.shape[1] != 3:
        raise ValueError(
            "detectable_los_unit_vectors must have shape (N, 3)."
        )
    if weights.shape != (los.shape[0],):
        raise ValueError(
            "detectable_residence_days must have shape (N,)."
        )

    scores = np.zeros(candidates.shape[0], dtype=float)
    cosine_fov = float(np.cos(fov_half_angle_rad))

    for candidate_start in range(
        0,
        candidates.shape[0],
        candidate_batch_size,
    ):
        candidate_stop = min(
            candidate_start + candidate_batch_size,
            candidates.shape[0],
        )
        candidate_block = candidates[candidate_start:candidate_stop]
        block_scores = np.zeros(candidate_block.shape[0], dtype=float)

        for los_start in range(0, los.shape[0], los_batch_size):
            los_stop = min(los_start + los_batch_size, los.shape[0])
            los_block = los[los_start:los_stop]
            weight_block = weights[los_start:los_stop]

            dot_products = candidate_block @ los_block.T
            inside = dot_products >= cosine_fov
            block_scores += inside @ weight_block

        scores[candidate_start:candidate_stop] = block_scores

    return scores


def score_single_boresight_with_local_snr(
    boresight_synodic: FloatArray,
    positions_synodic_km: FloatArray,
    residence_days: FloatArray,
    los_unit_vectors: FloatArray,
    snr_evaluator: SNREvaluator,
    config: PointingConfig,
) -> tuple[float, int, int]:
    """Score one boresight while evaluating SNR only inside its FOV.

    Returns
    -------
    score_days
        Detectable residence-time score.
    number_of_geometric_cells
        Number of residence cells geometrically inside the FOV.
    number_of_detectable_cells
        Number of those cells satisfying the SNR threshold.
    """

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
        snr_evaluator,
        config.snr_batch_size,
    )
    detectable = np.isfinite(snr) & (snr >= config.snr_threshold)
    score = float(np.sum(residence_days[selected_indices][detectable]))
    return score, int(selected_indices.size), int(np.count_nonzero(detectable))


# =============================================================================
# Production adapter to payload_asteroid_snr_model.py
# =============================================================================


@dataclass
class PayloadSNRGridEvaluator:
    """Adapter from synodic grid positions to the payload SNR model.

    The payload, asteroid, environment, and options objects must be instances
    from ``payload_asteroid_snr_model.py``. All supplied Sun/Earth/Moon and
    observer states are inertial and belong to the same epoch and origin.

    ``asteroid_velocity_inertial_km_s`` may be one vector with shape (3,),
    broadcast to every hypothetical grid location, or an array with shape
    (N, 3) only when calling this evaluator directly with the matching N.
    A scalar angular-rate override may be supplied when a position-dependent
    inertial velocity is unavailable.
    """

    frame_transform: SynodicInertialTransform
    payload: Any
    asteroid: Any

    observer_position_inertial_km: ArrayLike
    observer_velocity_inertial_km_s: ArrayLike
    asteroid_velocity_inertial_km_s: ArrayLike

    sun_position_inertial_km: ArrayLike
    earth_position_inertial_km: ArrayLike
    moon_position_inertial_km: ArrayLike

    environment: Any = None
    options: Any = None
    phase_model: Any = None
    asteroid_angular_rate_override_arcsec_s: float | None = None

    def __post_init__(self) -> None:
        vector_fields = {
            "observer_position_inertial_km": (
                self.observer_position_inertial_km
            ),
            "observer_velocity_inertial_km_s": (
                self.observer_velocity_inertial_km_s
            ),
            "sun_position_inertial_km": self.sun_position_inertial_km,
            "earth_position_inertial_km": self.earth_position_inertial_km,
            "moon_position_inertial_km": self.moon_position_inertial_km,
        }
        for name, value in vector_fields.items():
            array = np.asarray(value, dtype=float)
            if array.shape != (3,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite vector of shape (3,).")
            setattr(self, name, array)

        velocity = np.asarray(
            self.asteroid_velocity_inertial_km_s,
            dtype=float,
        )
        if velocity.shape != (3,) and not (
            velocity.ndim == 2 and velocity.shape[1] == 3
        ):
            raise ValueError(
                "asteroid_velocity_inertial_km_s must have shape (3,) or "
                "shape (N, 3)."
            )
        self.asteroid_velocity_inertial_km_s = velocity

    def __call__(self, positions_synodic_km: FloatArray) -> FloatArray:
        try:
            from payload_asteroid_snr_model import (
                EnvironmentConfig,
                HG12PhaseModel,
                ObservationGeometry,
                SNROptions,
                compute_asteroid_snr,
            )
        except ImportError as exc:
            raise ImportError(
                "PayloadSNRGridEvaluator requires "
                "payload_asteroid_snr_model.py on the Python path."
            ) from exc

        positions_synodic = np.asarray(positions_synodic_km, dtype=float)
        positions_inertial = self.frame_transform.positions_synodic_to_inertial(
            positions_synodic
        )

        relative_inertial = (
            positions_inertial
            - np.asarray(self.observer_position_inertial_km)[None, :]
        )
        ranges = np.linalg.norm(relative_inertial, axis=1)
        if np.any(ranges <= 0.0):
            raise ValueError(
                "A residence cell coincides with the observer position."
            )
        point_to_grid_boresights = relative_inertial / ranges[:, None]

        asteroid_velocity = np.asarray(
            self.asteroid_velocity_inertial_km_s,
            dtype=float,
        )
        if asteroid_velocity.shape == (3,):
            asteroid_velocity = np.broadcast_to(
                asteroid_velocity,
                positions_inertial.shape,
            )
        elif asteroid_velocity.shape != positions_inertial.shape:
            raise ValueError(
                "A batched asteroid_velocity_inertial_km_s must match the "
                "positions passed to this evaluator. For optimizer batching, "
                "use one broadcast velocity vector or provide a custom SNR "
                "callback that indexes a full velocity field."
            )

        geometry = ObservationGeometry(
            observer_position_km=np.asarray(
                self.observer_position_inertial_km,
            ),
            observer_velocity_km_s=np.asarray(
                self.observer_velocity_inertial_km_s,
            ),
            asteroid_position_km=positions_inertial,
            asteroid_velocity_km_s=asteroid_velocity,
            sun_position_km=np.asarray(self.sun_position_inertial_km),
            earth_position_km=np.asarray(self.earth_position_inertial_km),
            moon_position_km=np.asarray(self.moon_position_inertial_km),
            boresight_unit_vector=point_to_grid_boresights,
            asteroid_angular_rate_arcsec_s=(
                self.asteroid_angular_rate_override_arcsec_s
            ),
        )

        result = compute_asteroid_snr(
            payload=self.payload,
            asteroid=self.asteroid,
            geometry=geometry,
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
            phase_model=(
                self.phase_model
                if self.phase_model is not None
                else HG12PhaseModel.from_default_table()
            ),
        )
        return np.asarray(result.snr, dtype=float).reshape(-1)


# =============================================================================
# Main optimizer
# =============================================================================


def determine_search_boresight(
    residence_grid: ResidenceGrid,
    observer_position_synodic_km: ArrayLike,
    previous_boresight_inertial: ArrayLike,
    previous_residence_score_days: float | None,
    invisibility_zone_axis_synodic: ArrayLike,
    frame_transform: SynodicInertialTransform,
    snr_evaluator: SNREvaluator,
    config: PointingConfig,
) -> PointingResult:
    """Determine one hard-constrained discrete search boresight."""

    positions = np.asarray(
        residence_grid.positions_synodic_km,
        dtype=float,
    )
    residence_days = np.asarray(
        residence_grid.residence_time_days,
        dtype=float,
    )

    previous_boresight_i = _unit_vector(
        previous_boresight_inertial,
        "previous_boresight_inertial",
    )
    reference_synodic = _unit_vector(
        frame_transform.directions_inertial_to_synodic(
            previous_boresight_i,
        ),
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
        & (residence_days > 0.0)
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

    # Optional computational shortcut. It is attempted only when the previous
    # score is positive and the complete current FOV remains IZ-feasible.
    shortcut_eligible = (
        config.enable_previous_score_shortcut
        and previous_residence_score_days is not None
        and np.isfinite(previous_residence_score_days)
        and previous_residence_score_days > 0.0
        and single_boresight_is_iz_feasible(
            reference_synodic,
            iz_axis_synodic,
            config,
        )
    )

    if shortcut_eligible:
        current_score, _, n_current_detectable = (
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
                frame_transform.directions_synodic_to_inertial(
                    reference_synodic,
                ),
                "retained_boresight_inertial",
            )
            return PointingResult(
                status=(
                    PointingStatus.SUCCESS_CURRENT_BORESIGHT_RETAINED
                ),
                boresight_synodic=reference_synodic,
                boresight_inertial=retained_inertial,
                residence_score_days=current_score,
                retained_current_boresight=True,
                number_of_residence_cells=number_of_residence_cells,
                # The shortcut deliberately does not evaluate SNR outside
                # the retained FOV, so the full-grid detectable count is
                # unknown. The selected-FOV count remains available.
                number_of_detectable_cells=None,
                number_of_candidate_boresights=0,
                number_of_feasible_boresights=0,
                number_of_cells_in_selected_fov=n_current_detectable,
                reference_boresight_synodic=reference_synodic,
            )

    # Full-grid deterministic SNR mask.
    snr_values = evaluate_snr_in_batches(
        positions,
        snr_evaluator,
        config.snr_batch_size,
    )
    detectable = (
        np.isfinite(snr_values)
        & (snr_values >= config.snr_threshold)
    )
    n_detectable = int(np.count_nonzero(detectable))

    if n_detectable == 0:
        return PointingResult(
            status=PointingStatus.NO_DETECTABLE_RESIDENCE_CELLS,
            boresight_synodic=None,
            boresight_inertial=None,
            residence_score_days=0.0,
            retained_current_boresight=False,
            number_of_residence_cells=number_of_residence_cells,
            number_of_detectable_cells=0,
            number_of_candidate_boresights=None,
            number_of_feasible_boresights=None,
            number_of_cells_in_selected_fov=0,
            reference_boresight_synodic=reference_synodic,
        )

    detectable_los = los_all[detectable]
    detectable_residence = residence_days[detectable]

    # The maximum boresight-change constraint is embedded in generation.
    candidates = generate_spherical_cap_candidates(
        reference_boresight_unit=reference_synodic,
        maximum_angle_rad=config.maximum_boresight_change_rad,
        angular_spacing_rad=config.candidate_angular_spacing_rad,
    )
    n_candidates = int(candidates.shape[0])

    iz_feasible = invisibility_zone_feasible_mask(
        candidate_boresights_synodic=candidates,
        invisibility_zone_axis_synodic=iz_axis_synodic,
        invisibility_zone_half_angle_rad=(
            config.invisibility_zone_half_angle_rad
        ),
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
            number_of_detectable_cells=n_detectable,
            number_of_candidate_boresights=n_candidates,
            number_of_feasible_boresights=0,
            number_of_cells_in_selected_fov=0,
            reference_boresight_synodic=reference_synodic,
        )

    scores = score_candidates_batched(
        candidate_boresights=feasible_candidates,
        detectable_los_unit_vectors=detectable_los,
        detectable_residence_days=detectable_residence,
        fov_half_angle_rad=config.fov_half_angle_rad,
        candidate_batch_size=config.candidate_batch_size,
        los_batch_size=config.los_batch_size,
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
            number_of_detectable_cells=n_detectable,
            number_of_candidate_boresights=n_candidates,
            number_of_feasible_boresights=n_feasible,
            number_of_cells_in_selected_fov=0,
            reference_boresight_synodic=reference_synodic,
        )

    selected_synodic = _unit_vector(
        feasible_candidates[best_index],
        "selected_boresight_synodic",
    )
    selected_inertial = _unit_vector(
        frame_transform.directions_synodic_to_inertial(
            selected_synodic,
        ),
        "selected_boresight_inertial",
    )

    selected_membership = fov_membership_mask(
        selected_synodic,
        detectable_los,
        config.fov_half_angle_rad,
    )

    return PointingResult(
        status=PointingStatus.SUCCESS_OPTIMIZED_BORESIGHT,
        boresight_synodic=selected_synodic,
        boresight_inertial=selected_inertial,
        residence_score_days=best_score,
        retained_current_boresight=False,
        number_of_residence_cells=number_of_residence_cells,
        number_of_detectable_cells=n_detectable,
        number_of_candidate_boresights=n_candidates,
        number_of_feasible_boresights=n_feasible,
        number_of_cells_in_selected_fov=int(
            np.count_nonzero(selected_membership)
        ),
        reference_boresight_synodic=reference_synodic,
    )


# =============================================================================
# Minimum working example
# =============================================================================


def _direction_from_angles(
    off_axis_angle_rad: float,
    azimuth_rad: float,
) -> FloatArray:
    """Direction measured from +x, with azimuth in the y-z plane."""

    return np.array(
        [
            np.cos(off_axis_angle_rad),
            np.sin(off_axis_angle_rad) * np.cos(azimuth_rad),
            np.sin(off_axis_angle_rad) * np.sin(azimuth_rad),
        ],
        dtype=float,
    )


def _make_demo_residence_csv(path: Path) -> None:
    """Create two off-axis residence lobes for a self-contained example."""

    rng = np.random.default_rng(17)
    observer = np.array([-1.50e6, 0.0, 0.0], dtype=float)

    rows: list[dict[str, float]] = []
    lobe_definitions = [
        # Stronger lobe above the Earth direction.
        (np.deg2rad(28.0), np.deg2rad(0.0), 90, 9.0),
        # Weaker lobe on the opposite side.
        (np.deg2rad(35.0), np.deg2rad(180.0), 70, 5.5),
    ]

    for central_angle, central_azimuth, count, mean_weight in lobe_definitions:
        for _ in range(count):
            angle = central_angle + rng.normal(0.0, np.deg2rad(2.0))
            azimuth = central_azimuth + rng.normal(0.0, np.deg2rad(4.0))
            direction = _direction_from_angles(angle, azimuth)
            distance = rng.uniform(0.45e6, 2.4e6)
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


def _demo_snr_evaluator_factory(
    observer_position_synodic_km: FloatArray,
) -> SNREvaluator:
    """Return a deterministic demo SNR map for the self-test.

    This function exists only so the file runs without external mission files.
    Replace it in production with PayloadSNRGridEvaluator or another callable
    that evaluates the real SNR map.
    """

    observer = np.asarray(observer_position_synodic_km, dtype=float)

    def evaluate(positions_synodic_km: FloatArray) -> FloatArray:
        positions = np.asarray(positions_synodic_km, dtype=float)
        relative = positions - observer[None, :]
        distance = np.linalg.norm(relative, axis=1)
        los = relative / distance[:, None]

        # Deterministic range response plus a mild one-sided phase-like factor.
        range_term = 16.0 * np.exp(-distance / 2.8e6)
        side_factor = np.where(los[:, 1] >= 0.0, 1.15, 0.82)
        return np.asarray(range_term * side_factor, dtype=float)

    return evaluate


def minimum_working_example() -> None:
    """Run and assert both the full-search and shortcut-retention paths."""

    observer_synodic_km = np.array([-1.50e6, 0.0, 0.0], dtype=float)

    # Identity is sufficient for this synthetic example. In production,
    # provide the current epoch's true R_I_from_S and Earth inertial position.
    frame_transform = SynodicInertialTransform(
        rotation_inertial_from_synodic=np.eye(3),
        synodic_origin_position_inertial_km=np.zeros(3),
    )

    # Earth/invisibility direction from the observer is +x in this example.
    iz_axis_synodic = np.array([1.0, 0.0, 0.0], dtype=float)

    # Initial payload direction is already near the stronger residence lobe.
    initial_boresight_inertial = _direction_from_angles(
        np.deg2rad(25.0),
        0.0,
    )

    config = PointingConfig(
        snr_threshold=5.0,
        fov_half_angle_rad=np.deg2rad(6.0),
        maximum_boresight_change_rad=np.deg2rad(90.0),
        candidate_angular_spacing_rad=np.deg2rad(1.0),
        invisibility_zone_half_angle_rad=np.deg2rad(10.0),
        invisibility_zone_margin_rad=np.deg2rad(1.0),
        enable_previous_score_shortcut=True,
        shortcut_minimum_previous_score_fraction=0.95,
        snr_batch_size=1_000,
        candidate_batch_size=64,
        los_batch_size=2_000,
    )

    snr_evaluator = _demo_snr_evaluator_factory(observer_synodic_km)

    with TemporaryDirectory() as temporary_directory:
        residence_csv = Path(temporary_directory) / "demo_residence_grid.csv"
        _make_demo_residence_csv(residence_csv)
        residence_grid = load_sparse_residence_grid_csv(residence_csv)

        # First call: no previous score, so the full discrete search must run.
        first = determine_search_boresight(
            residence_grid=residence_grid,
            observer_position_synodic_km=observer_synodic_km,
            previous_boresight_inertial=initial_boresight_inertial,
            previous_residence_score_days=None,
            invisibility_zone_axis_synodic=iz_axis_synodic,
            frame_transform=frame_transform,
            snr_evaluator=snr_evaluator,
            config=config,
        )

        assert first.status == PointingStatus.SUCCESS_OPTIMIZED_BORESIGHT
        assert first.boresight_inertial is not None
        assert first.residence_score_days > 0.0
        assert single_boresight_is_iz_feasible(
            first.boresight_synodic,
            iz_axis_synodic,
            config,
        )

        # Second call: unchanged geometry and the previous optimum should pass
        # the 95%-of-previous-score shortcut without generating candidates.
        second = determine_search_boresight(
            residence_grid=residence_grid,
            observer_position_synodic_km=observer_synodic_km,
            previous_boresight_inertial=first.boresight_inertial,
            previous_residence_score_days=first.residence_score_days,
            invisibility_zone_axis_synodic=iz_axis_synodic,
            frame_transform=frame_transform,
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

    print("Minimum working example completed successfully.")
    print(f"First status:  {first.status.value}")
    print(f"First score:   {first.residence_score_days:.3f} days")
    print(
        "First boresight (synodic): "
        f"{np.array2string(first.boresight_synodic, precision=6)}"
    )
    print(f"Second status: {second.status.value}")
    print(f"Second score:  {second.residence_score_days:.3f} days")


if __name__ == "__main__":
    minimum_working_example()
