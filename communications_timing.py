from __future__ import annotations

"""Geometry-driven mothership/microsatellite crosslink timing model.

The model assumes the mission architecture discussed for the centralized
constellation:

* one body-fixed directional crosslink antenna on the mothership;
* broad-beam/fixed microsatellite crosslink antennas;
* every communication sequence starts with the mothership Earth-pointing;
* contacts proceed in deployment order, i.e. increasing phase distance behind
  the mothership (nearest-deployed microsatellite first);
* fixed crosslink bit rate; no link-budget-dependent adaptive rate;
* one-way propagation delay is serialized conservatively into each contact;
* slew timing uses the inverse of the same bang-off-bang envelope used by the
  attitude-coordination code.

All Cartesian states are expected in one common Earth-centered inertial frame,
with position in kilometres.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


SPEED_OF_LIGHT_KM_S = 299792.458


def _unit(vector, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-15:
        raise ValueError(f"{name} must have a finite non-zero norm.")
    return value / norm


def _angle_between(u, v) -> float:
    u_hat = _unit(u, "first direction")
    v_hat = _unit(v, "second direction")
    return float(np.arccos(np.clip(np.dot(u_hat, v_hat), -1.0, 1.0)))


def bang_off_bang_slew_time(theta_rad: float, alpha_max_rad_s2: float, omega_max_rad_s: float) -> float:
    """Return minimum slew time for the existing bang-off-bang slew envelope.

    This is the analytical inverse of ``theta_s_of_dt`` in
    ``od_attcoord_coverage_mode_v2_tracking_anchor_dynamic_iv.py``:

        theta = alpha * dt^2 / 4                      (triangular profile)
        theta = (dt - omega/alpha) * omega            (rate-limited profile)

    with transition angle ``omega^2 / alpha``.
    """
    theta = float(theta_rad)
    alpha = float(alpha_max_rad_s2)
    omega = float(omega_max_rad_s)

    if not np.isfinite(theta) or theta < 0.0:
        raise ValueError("theta_rad must be finite and non-negative.")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha_max_rad_s2 must be finite and positive.")
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega_max_rad_s must be finite and positive.")

    theta_crit = omega * omega / alpha
    if theta <= theta_crit:
        return float(2.0 * np.sqrt(theta / alpha))
    return float(theta / omega + omega / alpha)


@dataclass(frozen=True)
class ContactTiming:
    stage: str
    contact_sequence: int
    spacecraft_index: int
    spacecraft_id: int
    spacecraft_phase_deg: float
    phase_distance_behind_mothership_deg: float
    range_km: float
    slew_angle_rad: float
    slew_time_sec: float
    settle_time_sec: float
    acquisition_time_sec: float
    packet_size_bits: float
    bitrate_bps: float
    tx_time_sec: float
    propagation_time_sec: float
    total_contact_time_sec: float

    @property
    def slew_angle_deg(self) -> float:
        return float(np.rad2deg(self.slew_angle_rad))


@dataclass(frozen=True)
class SequenceTiming:
    stage: str
    contacts: tuple[ContactTiming, ...]

    @property
    def total_time_sec(self) -> float:
        return float(sum(contact.total_contact_time_sec for contact in self.contacts))

    @property
    def total_slew_time_sec(self) -> float:
        return float(sum(contact.slew_time_sec for contact in self.contacts))

    @property
    def total_settle_time_sec(self) -> float:
        return float(sum(contact.settle_time_sec for contact in self.contacts))

    @property
    def total_acquisition_time_sec(self) -> float:
        return float(sum(contact.acquisition_time_sec for contact in self.contacts))

    @property
    def total_tx_time_sec(self) -> float:
        return float(sum(contact.tx_time_sec for contact in self.contacts))

    @property
    def total_propagation_time_sec(self) -> float:
        return float(sum(contact.propagation_time_sec for contact in self.contacts))

    @property
    def max_range_km(self) -> float:
        if not self.contacts:
            return float("nan")
        return float(max(contact.range_km for contact in self.contacts))

    @property
    def max_slew_angle_deg(self) -> float:
        if not self.contacts:
            return float("nan")
        return float(max(contact.slew_angle_deg for contact in self.contacts))

    @property
    def contact_order_ids(self) -> tuple[int, ...]:
        return tuple(contact.spacecraft_id for contact in self.contacts)


@dataclass(frozen=True)
class CommunicationCyclePlan:
    measurement: SequenceTiming
    command: SequenceTiming

    def outer_summary(self, *, include_measurement: bool, include_command: bool) -> Dict[str, object]:
        sequences: List[SequenceTiming] = []
        if include_measurement:
            sequences.append(self.measurement)
        if include_command:
            sequences.append(self.command)

        measurement_time = self.measurement.total_time_sec if include_measurement else 0.0
        command_time = self.command.total_time_sec if include_command else 0.0

        contacts = [contact for sequence in sequences for contact in sequence.contacts]
        return {
            "measurement_comm_time_sec": float(measurement_time),
            "command_comm_time_sec": float(command_time),
            "comm_total_time_sec": float(measurement_time + command_time),
            "comm_total_slew_time_sec": float(sum(c.slew_time_sec for c in contacts)),
            "comm_total_settle_time_sec": float(sum(c.settle_time_sec for c in contacts)),
            "comm_total_acquisition_time_sec": float(sum(c.acquisition_time_sec for c in contacts)),
            "comm_total_tx_time_sec": float(sum(c.tx_time_sec for c in contacts)),
            "comm_total_propagation_time_sec": float(sum(c.propagation_time_sec for c in contacts)),
            "comm_max_range_km": (
                float(max(c.range_km for c in contacts)) if contacts else np.nan
            ),
            "comm_max_slew_angle_deg": (
                float(max(c.slew_angle_deg for c in contacts)) if contacts else np.nan
            ),
            "comm_contact_order": ";".join(
                str(spacecraft_id) for spacecraft_id in self.command.contact_order_ids
            ),
        }

    def diagnostic_rows(self, *, include_measurement: bool, include_command: bool) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        sequences: Iterable[SequenceTiming] = (
            sequence
            for sequence, include in (
                (self.measurement, include_measurement),
                (self.command, include_command),
            )
            if include
        )
        for sequence in sequences:
            for contact in sequence.contacts:
                rows.append({
                    "stage": contact.stage,
                    "contact_sequence": int(contact.contact_sequence),
                    "spacecraft_index": int(contact.spacecraft_index),
                    "spacecraft_id": int(contact.spacecraft_id),
                    "spacecraft_phase_deg": float(contact.spacecraft_phase_deg),
                    "phase_distance_behind_mothership_deg": float(
                        contact.phase_distance_behind_mothership_deg
                    ),
                    "range_km": float(contact.range_km),
                    "slew_angle_deg": float(contact.slew_angle_deg),
                    "slew_time_sec": float(contact.slew_time_sec),
                    "settle_time_sec": float(contact.settle_time_sec),
                    "acquisition_time_sec": float(contact.acquisition_time_sec),
                    "packet_size_bits": float(contact.packet_size_bits),
                    "bitrate_bps": float(contact.bitrate_bps),
                    "tx_time_sec": float(contact.tx_time_sec),
                    "propagation_time_sec": float(contact.propagation_time_sec),
                    "total_contact_time_sec": float(contact.total_contact_time_sec),
                })
        return rows


class CommunicationsTimingModel:
    """Calculate one measurement and command crosslink plan for an OD cycle."""

    def __init__(self, config):
        communications = config.get("communications", {}) or {}
        self.enabled = bool(communications.get("enabled", False))
        self.diagnostics_enabled = bool(
            (communications.get("diagnostics", {}) or {}).get("enabled", True)
        )

        if not self.enabled:
            return

        crosslink = communications["crosslink"]
        self.bitrate_bps = float(crosslink["bitrate_bps"])
        self.measurement_packet_size_bits = float(
            crosslink["measurement_packet_size_bits"]
        )
        self.command_packet_size_bits = float(crosslink["command_packet_size_bits"])
        self.settle_time_sec = float(crosslink["settle_time_sec"])
        self.acquisition_time_sec = float(crosslink["acquisition_time_sec"])

        mothership = config["mothership"]
        derived = mothership["derived"]
        self.alpha_max_rad_s2 = float(derived["alpha_max_rad_s2"])
        self.omega_max_rad_s = float(derived["omega_max_rad_s"])

        phasing = config.get("formation_phasing", {}) or {}
        mode = str(phasing.get("mode", "equal")).strip().lower()
        num_sc = int(config["num_spacecraft"])
        if mode == "equal":
            self.spacecraft_phases_deg = np.linspace(
                0.0, 360.0, num_sc, endpoint=False, dtype=float
            )
        else:
            self.spacecraft_phases_deg = np.asarray(
                phasing["relative_phase_deg"], dtype=float
            ).reshape(-1)

        self.mothership_phase_deg = float(phasing["mothership_phase_deg"]) % 360.0
        self.phase_distance_behind_mothership_deg = np.mod(
            self.mothership_phase_deg - self.spacecraft_phases_deg,
            360.0,
        )
        # Sequential deployment convention: the spacecraft immediately behind
        # the mothership is contacted first, then progressively earlier drops.
        self.contact_order = np.argsort(
            self.phase_distance_behind_mothership_deg,
            kind="stable",
        ).astype(int)

    def _geometry_contacts(self, mothership_position_km, spacecraft_positions_km):
        mothership_position = np.asarray(mothership_position_km, dtype=float).reshape(3)
        spacecraft_positions = np.asarray(spacecraft_positions_km, dtype=float)
        if spacecraft_positions.shape != (self.spacecraft_phases_deg.size, 3):
            raise ValueError(
                "spacecraft_positions_km must have shape "
                f"({self.spacecraft_phases_deg.size}, 3), got {spacecraft_positions.shape}."
            )
        if np.any(~np.isfinite(mothership_position)) or np.any(~np.isfinite(spacecraft_positions)):
            raise ValueError("Crosslink geometry contains non-finite positions.")

        # Earth is the origin in the geocentric EME frame used by the OD loop.
        current_direction = _unit(-mothership_position, "mothership-to-Earth direction")
        geometry = []

        for contact_sequence, sc_index in enumerate(self.contact_order, start=1):
            relative = spacecraft_positions[int(sc_index)] - mothership_position
            range_km = float(np.linalg.norm(relative))
            if not np.isfinite(range_km) or range_km <= 0.0:
                raise ValueError(
                    f"Invalid mothership-to-SC{int(sc_index) + 1} range: {range_km}."
                )
            los = relative / range_km
            slew_angle_rad = _angle_between(current_direction, los)
            slew_time_sec = bang_off_bang_slew_time(
                slew_angle_rad,
                self.alpha_max_rad_s2,
                self.omega_max_rad_s,
            )
            geometry.append({
                "contact_sequence": int(contact_sequence),
                "spacecraft_index": int(sc_index),
                "spacecraft_id": int(sc_index) + 1,
                "spacecraft_phase_deg": float(self.spacecraft_phases_deg[int(sc_index)]),
                "phase_distance_behind_mothership_deg": float(
                    self.phase_distance_behind_mothership_deg[int(sc_index)]
                ),
                "range_km": range_km,
                "slew_angle_rad": slew_angle_rad,
                "slew_time_sec": slew_time_sec,
            })
            current_direction = los

        return geometry

    def _build_sequence(self, stage: str, packet_size_bits: float, geometry) -> SequenceTiming:
        tx_time_sec = float(packet_size_bits / self.bitrate_bps)
        contacts = []
        for item in geometry:
            propagation_time_sec = float(item["range_km"] / SPEED_OF_LIGHT_KM_S)
            total_contact_time_sec = (
                float(item["slew_time_sec"])
                + self.settle_time_sec
                + self.acquisition_time_sec
                + tx_time_sec
                + propagation_time_sec
            )
            contacts.append(ContactTiming(
                stage=str(stage),
                contact_sequence=int(item["contact_sequence"]),
                spacecraft_index=int(item["spacecraft_index"]),
                spacecraft_id=int(item["spacecraft_id"]),
                spacecraft_phase_deg=float(item["spacecraft_phase_deg"]),
                phase_distance_behind_mothership_deg=float(
                    item["phase_distance_behind_mothership_deg"]
                ),
                range_km=float(item["range_km"]),
                slew_angle_rad=float(item["slew_angle_rad"]),
                slew_time_sec=float(item["slew_time_sec"]),
                settle_time_sec=self.settle_time_sec,
                acquisition_time_sec=self.acquisition_time_sec,
                packet_size_bits=float(packet_size_bits),
                bitrate_bps=self.bitrate_bps,
                tx_time_sec=tx_time_sec,
                propagation_time_sec=propagation_time_sec,
                total_contact_time_sec=total_contact_time_sec,
            ))
        return SequenceTiming(stage=str(stage), contacts=tuple(contacts))

    def evaluate_cycle(self, mothership_state_eme_kms, spacecraft_states_eme_kms) -> CommunicationCyclePlan:
        if not self.enabled:
            raise RuntimeError("CommunicationsTimingModel is disabled in the configuration.")

        mothership_state = np.asarray(mothership_state_eme_kms, dtype=float).reshape(6)
        spacecraft_states = np.asarray(spacecraft_states_eme_kms, dtype=float)
        if spacecraft_states.shape != (self.spacecraft_phases_deg.size, 6):
            raise ValueError(
                "spacecraft_states_eme_kms must have shape "
                f"({self.spacecraft_phases_deg.size}, 6), got {spacecraft_states.shape}."
            )

        geometry = self._geometry_contacts(
            mothership_state[:3],
            spacecraft_states[:, :3],
        )
        return CommunicationCyclePlan(
            measurement=self._build_sequence(
                "measurement",
                self.measurement_packet_size_bits,
                geometry,
            ),
            command=self._build_sequence(
                "command",
                self.command_packet_size_bits,
                geometry,
            ),
        )
