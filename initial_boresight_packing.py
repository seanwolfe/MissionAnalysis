from __future__ import annotations

"""Epochwise initial-search boresight packing for a phased spacecraft formation.

The planner operates directly on the already matched SECR spacecraft
trajectories used by the overall detection simulation.  At every matched epoch,
each spacecraft receives its own tangent derived from its instantaneous
Earth--Moon invisibility-zone (IV-zone) axis.  Spacecraft are processed in fixed
ID order.  SC1 remains on the nominal boresight whenever possible and is shifted
outward only when required.  Subsequent spacecraft are packed inward while
feasible; after the first inward failure, all remaining spacecraft are packed
outward.

There is intentionally no fixed/global mode, no alternative nominal-violation
policy, and no visualization code in this production module.
"""

from dataclasses import dataclass

import numpy as np

from earth_moon_invisibility_zone import (
    compute_earth_moon_invisibility_zone_batch,
)


@dataclass(frozen=True)
class InitialBoresightSolution:
    """Complete initial-search boresight solution on the matched epoch grid."""

    boresights_secr: np.ndarray          # (E, S, 3)
    offsets_rad: np.ndarray              # (E, S), signed from nominal
    iv_axes_secr: np.ndarray             # (E, S, 3)
    iv_clearance_rad: np.ndarray         # (E, S), full-FOV clearance
    nominal_clearance_rad: np.ndarray    # (E, S), before outward correction
    inward_count: np.ndarray             # (E,)
    fov_half_angle_rad: float
    protected_half_angle_rad: float
    required_pairwise_separation_rad: float


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-15:
        raise ValueError(f"{name} must be a finite nonzero vector.")
    return value / norm


def _normalize_rows(vectors: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(values, axis=-1)
    bad = (~np.isfinite(norms)) | (norms <= 1.0e-15)
    if np.any(bad):
        raise ValueError(
            f"{name} contains invalid vectors at {np.argwhere(bad).tolist()}."
        )
    return values / norms[..., None]


def fov_area_deg2_to_half_angle_rad(fov_deg2: float) -> float:
    """Convert sky-area FOV in deg^2 to an equivalent circular half-angle."""

    area = float(fov_deg2)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError(f"fov must be finite and positive, got {fov_deg2!r}.")
    steradians = area / (180.0 / np.pi) ** 2
    argument = 1.0 - steradians / (2.0 * np.pi)
    if not -1.0 <= argument <= 1.0:
        raise ValueError(f"fov={area} deg^2 is too large for a spherical cap.")
    return float(np.arccos(argument))


def _angular_separation_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_hat = _normalize_rows(np.asarray(a, dtype=float), "first direction")
    b_hat = _normalize_rows(np.asarray(b, dtype=float), "second direction")
    return np.arccos(np.clip(np.sum(a_hat * b_hat, axis=-1), -1.0, 1.0))


def _tangent_toward_axis(nominal: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Project an IV axis into the tangent plane at the nominal boresight."""

    b0 = _unit(nominal, "nominal boresight")
    a = _unit(axis, "IV axis")
    tangent = a - np.dot(a, b0) * b0
    if float(np.linalg.norm(tangent)) <= 1.0e-12:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(trial, b0))) > 0.95:
            trial = np.array([0.0, 0.0, 1.0], dtype=float)
        tangent = trial - np.dot(trial, b0) * b0
    return _unit(tangent, "IV-directed tangent")


def _boresight_on_tangent(
    nominal: np.ndarray,
    tangent: np.ndarray,
    signed_offset_rad: float,
) -> np.ndarray:
    """Rotate the nominal direction along one signed great-circle tangent."""

    b0 = _unit(nominal, "nominal boresight")
    tangent_hat = _unit(tangent, "tangent")
    if abs(float(np.dot(b0, tangent_hat))) > 1.0e-10:
        raise ValueError("nominal boresight and tangent must be perpendicular.")
    offset = float(signed_offset_rad)
    result = np.cos(offset) * b0 + np.sin(offset) * tangent_hat
    return _unit(result, "constructed boresight")


def _full_fov_clearance_rad(
    boresight: np.ndarray,
    iv_axis: np.ndarray,
    protected_half_angle_rad: float,
) -> float:
    centre_separation = float(
        _angular_separation_rad(
            np.asarray(boresight, dtype=float).reshape(1, 3),
            np.asarray(iv_axis, dtype=float).reshape(1, 3),
        )[0]
    )
    return centre_separation - float(protected_half_angle_rad)


def _minimum_outward_shift_rad(
    nominal: np.ndarray,
    iv_axis: np.ndarray,
    protected_half_angle_rad: float,
    boundary_buffer_rad: float,
) -> float:
    """Return the minimum non-positive offset that clears the complete FOV."""

    alpha = float(
        _angular_separation_rad(
            _unit(nominal, "nominal boresight").reshape(1, 3),
            _unit(iv_axis, "IV axis").reshape(1, 3),
        )[0]
    )
    if alpha >= float(protected_half_angle_rad):
        return 0.0
    return alpha - float(protected_half_angle_rad) - float(boundary_buffer_rad)


def _offset_clears_iv(
    nominal: np.ndarray,
    tangent: np.ndarray,
    offset_rad: float,
    iv_axis: np.ndarray,
    protected_half_angle_rad: float,
) -> tuple[bool, np.ndarray, float]:
    boresight = _boresight_on_tangent(nominal, tangent, offset_rad)
    clearance = _full_fov_clearance_rad(
        boresight,
        iv_axis,
        protected_half_angle_rad,
    )
    return clearance >= -1.0e-12, boresight, clearance


def _pairwise_clear(
    candidate: np.ndarray,
    assigned: list[np.ndarray],
    required_separation_rad: float,
) -> bool:
    if not assigned:
        return True
    existing = np.asarray(assigned, dtype=float)
    candidate_rows = np.broadcast_to(
        _unit(candidate, "candidate boresight").reshape(1, 3),
        existing.shape,
    )
    separations = _angular_separation_rad(candidate_rows, existing)
    return bool(
        np.all(separations >= float(required_separation_rad) - 1.0e-12)
    )


def _read_configuration(config: dict) -> dict:
    packing = config.get("initial_boresight_packing", {}) or {}
    if not isinstance(packing, dict):
        raise TypeError("initial_boresight_packing must be a YAML mapping.")

    ems = config.get("ems", {}) or {}
    if not isinstance(ems, dict):
        raise TypeError("ems must be a YAML mapping.")
    if "half_angle_deg" not in ems:
        raise KeyError("ems.half_angle_deg is required for boresight packing.")

    required = (
        "nominal_boresight_secr",
        "fov_separation_margin_deg",
        "outward_clearance_buffer_deg",
        "packing_search_step_deg",
        "max_outward_offset_deg",
    )
    missing = [key for key in required if key not in packing]
    if missing:
        raise KeyError(
            "initial_boresight_packing is missing required entries: "
            f"{missing}."
        )

    nominal = _unit(
        np.asarray(packing["nominal_boresight_secr"], dtype=float).reshape(3),
        "initial_boresight_packing.nominal_boresight_secr",
    )

    values = {
        "nominal": nominal,
        "iv_half_angle_deg": float(ems["half_angle_deg"]),
        "fov_deg2": float(config["fov"]),
        "iv_clearance_margin_deg": float(
            ems["fov_clearance_margin_deg"]
        ),
        "fov_separation_margin_deg": float(
            packing["fov_separation_margin_deg"]
        ),
        "outward_clearance_buffer_deg": float(
            packing["outward_clearance_buffer_deg"]
        ),
        "packing_search_step_deg": float(packing["packing_search_step_deg"]),
        "max_outward_offset_deg": float(packing["max_outward_offset_deg"]),
    }

    nonnegative = (
        "iv_clearance_margin_deg",
        "fov_separation_margin_deg",
        "outward_clearance_buffer_deg",
    )
    for key in nonnegative:
        if not np.isfinite(values[key]) or values[key] < 0.0:
            raise ValueError(f"{key} must be finite and nonnegative.")

    positive = ("packing_search_step_deg", "max_outward_offset_deg")
    for key in positive:
        if not np.isfinite(values[key]) or values[key] <= 0.0:
            raise ValueError(f"{key} must be finite and positive.")

    if not np.isfinite(values["iv_half_angle_deg"]) or not (
        0.0 <= values["iv_half_angle_deg"] < 180.0
    ):
        raise ValueError("ems.half_angle_deg must lie in [0, 180) deg.")

    return values


def compute_initial_boresight_history(
    spacecraft_positions_secr_km: np.ndarray,
    moon_positions_secr_km: np.ndarray,
    config: dict,
) -> InitialBoresightSolution:
    """Compute one boresight per spacecraft at every matched epoch.

    Parameters
    ----------
    spacecraft_positions_secr_km
        Array with shape ``(epochs, spacecraft, 3)`` containing the actual
        already-phased matched spacecraft positions.
    moon_positions_secr_km
        Common Moon history with shape ``(epochs, 3)``.  The same history is
        used for every spacecraft at a given epoch.
    config
        Mission YAML mapping.  The payload FOV is read from top-level ``fov``;
        the IV half-angle and common full-FOV clearance margin are read from
        ``ems``; packing-specific separation/search limits are read from
        ``initial_boresight_packing``.
    """

    sc_positions = np.asarray(spacecraft_positions_secr_km, dtype=float)
    moon_positions = np.asarray(moon_positions_secr_km, dtype=float)

    if sc_positions.ndim != 3 or sc_positions.shape[2] != 3:
        raise ValueError(
            "spacecraft_positions_secr_km must have shape (E,S,3), got "
            f"{sc_positions.shape}."
        )
    epochs, num_spacecraft, _ = sc_positions.shape
    if epochs <= 0 or num_spacecraft <= 0:
        raise ValueError("At least one epoch and one spacecraft are required.")
    if moon_positions.shape != (epochs, 3):
        raise ValueError(
            "moon_positions_secr_km must have shape "
            f"({epochs},3), got {moon_positions.shape}."
        )
    if not np.all(np.isfinite(sc_positions)) or not np.all(
        np.isfinite(moon_positions)
    ):
        raise ValueError("Spacecraft and Moon histories must be finite.")

    values = _read_configuration(config)
    nominal = values["nominal"]

    fov_half_angle_rad = fov_area_deg2_to_half_angle_rad(values["fov_deg2"])
    iv_half_angle_rad = float(np.deg2rad(values["iv_half_angle_deg"]))
    protected_half_angle_rad = (
        iv_half_angle_rad
        + fov_half_angle_rad
        + float(np.deg2rad(values["iv_clearance_margin_deg"]))
    )
    required_pairwise_separation_rad = (
        2.0 * fov_half_angle_rad
        + float(np.deg2rad(values["fov_separation_margin_deg"]))
    )
    boundary_buffer_rad = float(
        np.deg2rad(values["outward_clearance_buffer_deg"])
    )
    search_step_rad = float(np.deg2rad(values["packing_search_step_deg"]))
    max_outward_rad = float(np.deg2rad(values["max_outward_offset_deg"]))

    earth_positions = np.zeros((epochs, 3), dtype=float)
    iv_axes = np.empty((epochs, num_spacecraft, 3), dtype=float)
    for sc_index in range(num_spacecraft):
        zone = compute_earth_moon_invisibility_zone_batch(
            spacecraft_position=sc_positions[:, sc_index, :],
            earth_position=earth_positions,
            moon_position=moon_positions,
            half_angle_deg=values["iv_half_angle_deg"],
        )
        iv_axes[:, sc_index, :] = np.asarray(zone.axis_geo_eme, dtype=float)

    boresights = np.empty((epochs, num_spacecraft, 3), dtype=float)
    offsets = np.empty((epochs, num_spacecraft), dtype=float)
    clearance = np.empty((epochs, num_spacecraft), dtype=float)
    nominal_clearance = np.empty((epochs, num_spacecraft), dtype=float)
    inward_counts = np.empty(epochs, dtype=int)

    for epoch_index in range(epochs):
        epoch_axes = iv_axes[epoch_index]
        epoch_tangents = np.empty((num_spacecraft, 3), dtype=float)
        epoch_min_shifts = np.empty(num_spacecraft, dtype=float)

        for sc_index in range(num_spacecraft):
            axis = epoch_axes[sc_index]
            tangent = _tangent_toward_axis(nominal, axis)
            epoch_tangents[sc_index] = tangent
            nominal_clearance[epoch_index, sc_index] = _full_fov_clearance_rad(
                nominal,
                axis,
                protected_half_angle_rad,
            )
            epoch_min_shifts[sc_index] = _minimum_outward_shift_rad(
                nominal,
                axis,
                protected_half_angle_rad,
                boundary_buffer_rad,
            )

        assigned_vectors: list[np.ndarray] = []
        epoch_offsets = np.empty(num_spacecraft, dtype=float)
        epoch_boresights = np.empty((num_spacecraft, 3), dtype=float)
        epoch_clearance = np.empty(num_spacecraft, dtype=float)

        sc1_offset = (
            0.0
            if nominal_clearance[epoch_index, 0] >= -1.0e-12
            else float(epoch_min_shifts[0])
        )
        ok, sc1_boresight, sc1_clearance = _offset_clears_iv(
            nominal,
            epoch_tangents[0],
            sc1_offset,
            epoch_axes[0],
            protected_half_angle_rad,
        )
        if not ok:
            raise RuntimeError(
                f"SC1 outward correction failed at matched epoch {epoch_index}."
            )
        epoch_offsets[0] = sc1_offset
        epoch_boresights[0] = sc1_boresight
        epoch_clearance[0] = sc1_clearance
        assigned_vectors.append(sc1_boresight)

        inward_open = True
        inward_slot = 1
        outward_slot = 1
        inward_count = 0

        for sc_index in range(1, num_spacecraft):
            axis = epoch_axes[sc_index]
            tangent = epoch_tangents[sc_index]
            placed = False

            if inward_open:
                alpha = float(
                    _angular_separation_rad(
                        nominal.reshape(1, 3),
                        axis.reshape(1, 3),
                    )[0]
                )
                max_inward_offset = alpha - protected_half_angle_rad
                start_inward = inward_slot * required_pairwise_separation_rad

                if start_inward <= max_inward_offset + 1.0e-12:
                    candidate_offset = start_inward
                    while candidate_offset <= max_inward_offset + 1.0e-12:
                        ok, candidate_boresight, candidate_clearance = (
                            _offset_clears_iv(
                                nominal,
                                tangent,
                                candidate_offset,
                                axis,
                                protected_half_angle_rad,
                            )
                        )
                        if ok and _pairwise_clear(
                            candidate_boresight,
                            assigned_vectors,
                            required_pairwise_separation_rad,
                        ):
                            epoch_offsets[sc_index] = float(candidate_offset)
                            epoch_boresights[sc_index] = candidate_boresight
                            epoch_clearance[sc_index] = candidate_clearance
                            assigned_vectors.append(candidate_boresight)
                            inward_count += 1
                            inward_slot += 1
                            placed = True
                            break
                        candidate_offset += search_step_rad

                if not placed:
                    inward_open = False

            if not placed:
                desired_outward = (
                    -outward_slot * required_pairwise_separation_rad
                )
                start_outward = min(
                    desired_outward,
                    float(epoch_min_shifts[sc_index]),
                )
                candidate_offset = start_outward

                while candidate_offset >= -max_outward_rad - 1.0e-12:
                    ok, candidate_boresight, candidate_clearance = (
                        _offset_clears_iv(
                            nominal,
                            tangent,
                            candidate_offset,
                            axis,
                            protected_half_angle_rad,
                        )
                    )
                    if ok and _pairwise_clear(
                        candidate_boresight,
                        assigned_vectors,
                        required_pairwise_separation_rad,
                    ):
                        epoch_offsets[sc_index] = float(candidate_offset)
                        epoch_boresights[sc_index] = candidate_boresight
                        epoch_clearance[sc_index] = candidate_clearance
                        assigned_vectors.append(candidate_boresight)
                        outward_slot += 1
                        placed = True
                        break
                    candidate_offset -= search_step_rad

            if not placed:
                raise RuntimeError(
                    "No outward initial-boresight placement found at matched "
                    f"epoch {epoch_index} for SC{sc_index + 1} within "
                    f"{values['max_outward_offset_deg']:.6g} deg of nominal."
                )

        inward_counts[epoch_index] = inward_count
        offsets[epoch_index] = epoch_offsets
        boresights[epoch_index] = epoch_boresights
        clearance[epoch_index] = epoch_clearance

    return InitialBoresightSolution(
        boresights_secr=boresights,
        offsets_rad=offsets,
        iv_axes_secr=iv_axes,
        iv_clearance_rad=clearance,
        nominal_clearance_rad=nominal_clearance,
        inward_count=inward_counts,
        fov_half_angle_rad=fov_half_angle_rad,
        protected_half_angle_rad=protected_half_angle_rad,
        required_pairwise_separation_rad=required_pairwise_separation_rad,
    )
