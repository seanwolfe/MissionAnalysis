import pandas as pd
import numpy as np
import spiceypy as spice
from Spacecraft import Spacecraft

class Formation:
    def __init__(self, configs):
        prelim_orbit = pd.read_csv(configs['orbit_file_path'], sep=',', header=0, names=configs['orbit_column_names'])
        self.orbit = prelim_orbit.iloc[configs['quasi_halo_start']:configs['quasi_halo_end']]
        self.num_spacecraft = configs['num_spacecraft']
        self.sim_steps = None  # this comes from the asteroid trajectory
        self.spacecraft = None
        self.currently_detecting = None

        # Formation-phasing diagnostics. Indices are relative to the sliced
        # ``self.orbit`` dataframe and to the configured one-period interval.
        self.formation_anchor_index = None
        self.relative_phase_deg = None
        self.realized_relative_phase_deg = None
        self.phase_offset_indices = None
        self.spacecraft_initial_indices = None

        self.initial_formation(configs)

    def _quasi_halo_period_steps(self, configs):
        """Return and validate the number of samples in one halo period."""
        period_steps = int(
            configs['quasi_halo_one_period_end']
            - configs['quasi_halo_start']
        )
        if period_steps <= 0:
            raise ValueError(
                "quasi_halo_one_period_end must be greater than "
                "quasi_halo_start."
            )
        if period_steps > len(self.orbit):
            raise ValueError(
                "The configured one-period interval contains "
                f"{period_steps} samples, but the sliced orbit contains only "
                f"{len(self.orbit)} rows."
            )
        return period_steps

    def _relative_phase_degrees(self, configs):
        """Return requested spacecraft phases measured relative to SC1.

        ``formation_phasing.mode: equal`` is the default and produces
        ``[0, 360/N, ..., 360(N-1)/N]``. ``mode: custom`` reads
        ``relative_phase_deg`` from the same configuration block.
        """
        phasing = configs.get('formation_phasing', {}) or {}
        if not isinstance(phasing, dict):
            raise TypeError("formation_phasing must be a YAML mapping.")

        mode = str(phasing.get('mode', 'equal')).strip().lower()
        if mode == 'equal':
            phases = np.linspace(
                0.0,
                360.0,
                int(self.num_spacecraft),
                endpoint=False,
                dtype=float,
            )
        elif mode == 'custom':
            if 'relative_phase_deg' not in phasing:
                raise KeyError(
                    "formation_phasing.relative_phase_deg is required when "
                    "formation_phasing.mode is 'custom'."
                )
            phases = np.asarray(
                phasing['relative_phase_deg'], dtype=float
            ).reshape(-1)
            if phases.size != int(self.num_spacecraft):
                raise ValueError(
                    "formation_phasing.relative_phase_deg must contain exactly "
                    f"num_spacecraft={self.num_spacecraft} entries; received "
                    f"{phases.size}."
                )
        else:
            raise ValueError(
                "formation_phasing.mode must be 'equal' or 'custom', got "
                f"{mode!r}."
            )

        if np.any(~np.isfinite(phases)):
            raise ValueError("All relative spacecraft phases must be finite.")

        phases = np.mod(phases, 360.0)
        phases[np.isclose(phases, 360.0, atol=1.0e-12)] = 0.0

        if not np.isclose(phases[0], 0.0, atol=1.0e-12):
            raise ValueError(
                "The first relative phase must be 0 deg because all phases "
                "are defined relative to spacecraft 1."
            )

        sorted_phases = np.sort(phases)
        if sorted_phases.size > 1 and np.any(
            np.isclose(np.diff(sorted_phases), 0.0, atol=1.0e-12)
        ):
            raise ValueError(
                "Relative spacecraft phases must be distinct after wrapping "
                "into [0, 360) deg."
            )

        return phases

    def _build_formation_from_anchor(self, anchor_index, configs):
        """Construct the formation from an SC1 anchor and relative phases."""
        period_steps = self._quasi_halo_period_steps(configs)
        phases_deg = self._relative_phase_degrees(configs)

        anchor_index = int(anchor_index) % period_steps
        phase_offsets = np.rint(
            phases_deg / 360.0 * period_steps
        ).astype(np.int64)
        phase_offsets %= period_steps

        if np.unique(phase_offsets).size != phase_offsets.size:
            raise ValueError(
                "At least two requested phases map to the same LPF orbit row. "
                "Increase the phase separation or use a more finely sampled "
                "reference orbit."
            )

        scs_start = (
            anchor_index + phase_offsets
        ) % period_steps
        scs_start = scs_start.astype(np.int64)

        scs_ini_pos = [
            np.array(
                [
                    self.orbit['SUN_EARTH_CO_X_(km)'].iloc[int(sc_start)],
                    self.orbit['SUN_EARTH_CO_Y_(km)'].iloc[int(sc_start)],
                    self.orbit['SUN_EARTH_CO_Z_(km)'].iloc[int(sc_start)],
                ],
                dtype=float,
            )
            for sc_start in scs_start
        ]

        self.spacecraft = [
            Spacecraft(ini_pos, int(scs_start[k]), configs)
            for k, ini_pos in enumerate(scs_ini_pos)
        ]

        self.formation_anchor_index = int(anchor_index)
        self.relative_phase_deg = phases_deg.copy()
        self.phase_offset_indices = phase_offsets.copy()
        self.spacecraft_initial_indices = scs_start.copy()
        self.realized_relative_phase_deg = (
            phase_offsets.astype(float) / period_steps * 360.0
        )

    def initial_formation(self, configs):
        """Create a randomly anchored equal or custom-phased formation."""
        period_steps = self._quasi_halo_period_steps(configs)
        anchor_index = int(np.random.randint(0, period_steps))
        self._build_formation_from_anchor(anchor_index, configs)
        return

    def match_spacecraft_trajectory(self, asteroid_length, configs):
        """
        Resamples and aligns each spacecraft trajectory to match the asteroid trajectory's
        one-hour intervals and start time.

        Parameters:
            asteroid_times (pd.Series): Timestamps of the asteroid trajectory.
            configs (dict): Configuration dictionary with conversion factors.

        Returns:
            None (modifies each spacecraft's matched_trajectory in place).
        """

        for i, spacecraft in enumerate(self.spacecraft):
            # Convert spacecraft timestamps to pandas datetime format
            start_index = spacecraft.ini_pos_index  # Initial position index

            self.orbit['Time'] = pd.to_datetime(self.orbit['Time'])
            self.orbit = self.orbit.drop_duplicates(subset=['Time'])  # there are duplicates in the lisa pathfinder orbit file apparently

            # Assuming self.orbit['Time'] is already in datetime format
            original_timestamp = self.orbit.iloc[start_index]['Time']

            # Resample spacecraft data at hourly intervals (matching asteroid)
            spacecraft_resampled = self.orbit.set_index('Time').resample('h').nearest().reset_index()

            # Find the closest timestamp in the resampled data
            try:
                new_index = spacecraft_resampled[spacecraft_resampled['Time'] == original_timestamp].index[0]
            except IndexError:
                # If the exact timestamp is not found, use nearest method
                # Get the nearest index based on the original timestamp
                nearest_timestamp = spacecraft_resampled['Time'].iloc[
                    (spacecraft_resampled['Time'] - original_timestamp).abs().argmin()]
                new_index = spacecraft_resampled[spacecraft_resampled['Time'] == nearest_timestamp].index[0]

            # Keep only relevant position columns
            spacecraft_pos = spacecraft_resampled.loc[:, ['SUN_EARTH_CO_X_(km)',
                                                          'SUN_EARTH_CO_Y_(km)',
                                                          'SUN_EARTH_CO_Z_(km)']].to_numpy()

            sc_length = len(spacecraft_pos)

            # Create the trajectory starting at the correct index
            ordered_traj = np.vstack([spacecraft_pos[new_index:], spacecraft_pos[:new_index]])

            if sc_length >= asteroid_length:
                # Trim if spacecraft trajectory is longer
                adjusted_traj = ordered_traj[:asteroid_length]
            else:
                # Wrap around if spacecraft trajectory is shorter
                repeats = asteroid_length // sc_length
                remainder = asteroid_length % sc_length

                adjusted_traj = np.vstack([
                    np.tile(ordered_traj, (repeats, 1)),  # Full cycles
                    ordered_traj[:remainder]  # Remaining part
                ])

            # Convert to AU
            self.spacecraft[i].matched_trajectory = adjusted_traj / (
                configs['AU_TO_M'] / configs['KM_TO_M']
            )

        return

    def match_spacecraft_trajectory_full(self, asteroid_length, configs):
        """
        Resamples and aligns each spacecraft trajectory to match the asteroid trajectory's
        one-hour intervals and start time.

        TIME BEHAVIOR (as requested):
          - We resample onto an hourly grid using nearest-neighbor selection.
          - Instead of keeping the hourly grid times, we keep the ORIGINAL epoch of the
            row that was selected as nearest (by carrying a ROW_ID through resampling).

        Keeps ALL orbit columns.

        Unit conversion:
          - columns containing '(km)'   (but not '(km/s)') -> AU
          - columns containing '(km/s)'                    -> AU/day
          - Time stays datetime (original epochs)
        """

        AU_km = configs["AU_TO_M"] / configs["KM_TO_M"]
        SEC_PER_DAY = float(configs["SECONDS_PER_DAY"])

        if "Time" not in self.orbit.columns:
            raise ValueError("self.orbit must contain a 'Time' column")

        # ---- Clean orbit once ----
        orbit = self.orbit.copy()
        orbit["Time"] = pd.to_datetime(orbit["Time"])
        orbit = orbit.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)

        # Add a stable row id so we can recover the exact original epoch after resampling
        orbit["ROW_ID"] = np.arange(len(orbit), dtype=np.int64)

        # Cache a ROW_ID -> original Time mapping
        rowid_to_time = orbit.set_index("ROW_ID")["Time"]

        # All data columns except Time (we will handle Time explicitly)
        data_cols = [c for c in orbit.columns if c not in ("Time",)]
        # (data_cols includes ROW_ID; we will drop it at the end)

        # ---- Hourly resample using nearest original row ----
        # This returns a row whose values come from the nearest original timestamp;
        # crucially, ROW_ID comes along for the ride.
        spacecraft_resampled = (
            orbit.set_index("Time")
            .resample("h")
            .nearest()
            .reset_index()
        )

        # Replace resampled Time (hourly grid) with the ORIGINAL epoch of the selected row
        if "ROW_ID" not in spacecraft_resampled.columns:
            raise RuntimeError("ROW_ID missing after resample; cannot recover original epochs.")

        spacecraft_resampled["Time"] = spacecraft_resampled["ROW_ID"].map(rowid_to_time)

        # Optional: if nearest selection causes duplicates in Time, keep them (your choice).
        # If you want to drop duplicates after mapping, uncomment:
        # spacecraft_resampled = spacecraft_resampled.drop_duplicates(subset=["Time"]).reset_index(drop=True)

        # Columns we will carry into matched trajectory (everything except ROW_ID)
        out_cols = [c for c in spacecraft_resampled.columns if c != "ROW_ID"]
        numeric_cols = [c for c in out_cols if c != "Time"]  # convert these as needed

        for i, spacecraft in enumerate(self.spacecraft):
            start_index = spacecraft.ini_pos_index
            original_timestamp = orbit.iloc[start_index]["Time"]

            # Find the index in the resampled table whose ORIGINAL epoch matches the spacecraft start,
            # otherwise snap to nearest by time.
            idx_matches = spacecraft_resampled.index[spacecraft_resampled["Time"] == original_timestamp]
            if len(idx_matches) > 0:
                new_index = int(idx_matches[0])
            else:
                new_index = int((spacecraft_resampled["Time"] - original_timestamp).abs().argmin())

            sc_length = len(spacecraft_resampled)

            # Rotate the entire table (Time + all columns) together
            ordered_df = pd.concat(
                [spacecraft_resampled.iloc[new_index:], spacecraft_resampled.iloc[:new_index]],
                ignore_index=True,
            )

            # Keep only desired output columns (drops ROW_ID)
            ordered_df = ordered_df.loc[:, out_cols]

            # Trim or repeat to match asteroid_length
            if sc_length >= asteroid_length:
                adjusted_df = ordered_df.iloc[:asteroid_length].copy()
            else:
                repeats = asteroid_length // sc_length
                remainder = asteroid_length % sc_length
                adjusted_df = pd.concat(
                    [ordered_df] * repeats + [ordered_df.iloc[:remainder]],
                    ignore_index=True,
                )

            # --- Unit conversion ---
            # Velocities: km/s -> AU/day
            vel_cols = [c for c in numeric_cols if "(km/s)" in c]
            if vel_cols:
                adjusted_df[vel_cols] = adjusted_df[vel_cols] * (SEC_PER_DAY / AU_km)

            # Positions: km -> AU  (exclude km/s)
            pos_cols = [c for c in numeric_cols if "(km)" in c and "(km/s)" not in c]
            if pos_cols:
                adjusted_df[pos_cols] = adjusted_df[pos_cols] / AU_km

            self.spacecraft[i].matched_trajectory_full = adjusted_df

        # Write back cleaned orbit if you want side effects to persist
        self.orbit = orbit.drop(columns=["ROW_ID"])

        return

    def get_matched_spacecraft_positions_secr_km(self, configs):
        """Return actual matched spacecraft positions in SECR kilometres.

        ``match_spacecraft_trajectory_full`` stores position columns in AU even
        though their legacy column names retain ``(km)``.  This helper performs
        the conversion back to kilometres once and stacks the already-phased
        spacecraft trajectories as ``(epochs, spacecraft, 3)``.
        """
        if not self.spacecraft:
            raise RuntimeError("Formation contains no spacecraft.")

        columns = [
            'SUN_EARTH_CO_X_(km)',
            'SUN_EARTH_CO_Y_(km)',
            'SUN_EARTH_CO_Z_(km)',
        ]
        au_km = float(configs['AU_TO_M']) / float(configs['KM_TO_M'])
        histories = []
        expected_length = None

        for sc_index, spacecraft in enumerate(self.spacecraft):
            trajectory = spacecraft.matched_trajectory_full
            if trajectory is None:
                raise RuntimeError(
                    "match_spacecraft_trajectory_full must be called before "
                    "requesting matched SECR positions."
                )
            missing = [column for column in columns if column not in trajectory.columns]
            if missing:
                raise KeyError(
                    f"SC{sc_index + 1} matched trajectory is missing {missing}."
                )
            values = trajectory.loc[:, columns].to_numpy(dtype=float) * au_km
            if expected_length is None:
                expected_length = len(values)
            elif len(values) != expected_length:
                raise ValueError(
                    "All matched spacecraft trajectories must have equal length."
                )
            histories.append(values)

        return np.stack(histories, axis=1)

    def get_index_from_pos(self, position):
        possible_positions = self.orbit.loc[:, ['SUN_EARTH_CO_X_(km)', 'SUN_EARTH_CO_Y_(km)', 'SUN_EARTH_CO_Z_(km)']]
        distances = np.linalg.norm(possible_positions - position, axis=1)
        closest_position_idx = np.argmin(distances)
        return closest_position_idx

    def recall_formation(self, sc1_ini_index, config):
        """Rebuild a formation using the saved SC1 index and YAML phasing."""
        self._build_formation_from_anchor(sc1_ini_index, config)
        return

    def get_spacecraft_states(self, sc_ids=None):
        """
        Return spacecraft states in EME frame.

        Parameters
        ----------
        sc_ids : None, int, sequence of int, or boolean array
            - None: return all spacecraft states
            - int: return state of one spacecraft (shape: (6,))
            - sequence of int: return selected spacecraft states (N,6)
            - boolean array: mask over spacecraft list

        Returns
        -------
        np.ndarray
            Spacecraft state(s) in EME frame.
        """
        all_sc = self.spacecraft

        # Stack all states once
        states = np.vstack([sc.curr_state_eme for sc in all_sc])  # (M,6)

        if sc_ids is None:
            return states

        # Single spacecraft
        if isinstance(sc_ids, int):
            return states[sc_ids]

        # List / tuple / ndarray / boolean mask
        return states[sc_ids]

    def get_spacecraft_pointings(self, sc_ids=None):
        """
        Return spacecraft boresight vectors.

        Parameters
        ----------
        sc_ids : None, int, sequence of int, or boolean array
            - None: return all boresights (M,3)
            - int: return one boresight (3,)
            - sequence / mask: return selected boresights (N,3)

        Returns
        -------
        np.ndarray
            Spacecraft boresight vector(s).
        """
        # Stack once: (M,3)
        boresights = np.vstack([sc.boresight for sc in self.spacecraft])

        if sc_ids is None:
            return boresights

        if isinstance(sc_ids, int):
            return boresights[sc_ids]

        return boresights[sc_ids]

    def set_spacecraft_states(self, states_eme, sc_ids=None, *, set_epochs_from_row=None):
        """
        Set curr_state_eme for spacecraft (optionally a subset).

        Parameters
        ----------
        states_eme : np.ndarray
            If sc_ids is None: shape (M,6)
            If sc_ids is not None: shape (N,6) matching selected spacecraft count
        sc_ids : None, int, sequence of int, or boolean mask
            Which spacecraft to set.
        set_epochs_from_row : pandas.Series or None
            If provided, also sets sc.curr_sc_epoch from row["EPOCH_SC_i(jdtdb)"].
        """
        all_sc = self.spacecraft
        M = len(all_sc)

        if sc_ids is None:
            ids = np.arange(M)
        elif isinstance(sc_ids, int):
            ids = np.array([sc_ids], dtype=int)
            states_eme = np.atleast_2d(states_eme)
        else:
            ids = np.asarray(sc_ids)
            states_eme = np.asarray(states_eme)

        if states_eme.shape != (len(ids), 6):
            raise ValueError(f"states_eme must have shape ({len(ids)}, 6), got {states_eme.shape}")

        for j, sc_idx in enumerate(ids):
            sc = all_sc[int(sc_idx)]
            sc.curr_state_eme = states_eme[j, :]

            if set_epochs_from_row is not None:
                col = f"EPOCH_SC_{int(sc_idx) + 1}(jdtdb)"
                if col not in set_epochs_from_row.index:
                    raise KeyError(
                        f"Missing '{col}' in row. Available EPOCH_SC_* cols: "
                        f"{[c for c in set_epochs_from_row.index if str(c).startswith('EPOCH_SC_')]}"
                    )

                epoch_val = set_epochs_from_row[col]
                if pd.isna(epoch_val):
                    raise ValueError(f"'{col}' is NaN for spacecraft {int(sc_idx) + 1}")

                sc.curr_sc_epoch = float(epoch_val)

    def set_spacecraft_pointings(self, boresights_eme, sc_ids=None, *, normalize=True):
        """
        Set spacecraft boresight vectors.

        Parameters
        ----------
        boresights_eme : np.ndarray
            If sc_ids is None: shape (M,3)
            If sc_ids is not None: shape (N,3)
        sc_ids : None, int, sequence of int, or boolean mask
            Which spacecraft to set.
        normalize : bool
            If True, normalize boresights to unit vectors.
        """
        all_sc = self.spacecraft
        M = len(all_sc)

        if sc_ids is None:
            ids = np.arange(M)
        elif isinstance(sc_ids, int):
            ids = np.array([sc_ids], dtype=int)
            boresights_eme = np.atleast_2d(boresights_eme)
        else:
            ids = np.asarray(sc_ids)
            boresights_eme = np.asarray(boresights_eme)

        if boresights_eme.shape != (len(ids), 3):
            raise ValueError(f"boresights_eme must have shape ({len(ids)}, 3), got {boresights_eme.shape}")

        if normalize:
            norms = np.linalg.norm(boresights_eme, axis=1, keepdims=True)
            boresights_eme = boresights_eme / np.clip(norms, 1e-12, None)

        for j, sc_idx in enumerate(ids):
            all_sc[int(sc_idx)].boresight = boresights_eme[j, :]

    def detect(
            self,
            asteroid_state,
            epoch,
            n_body_prop,
            configs,
            *,
            moon_position_eme_km,
            absolute_magnitude_h=None,
            snr_evaluator=None,
    ):
        """Generate one OD tracklet after single-epoch geometry/SNR gating.

        Geometry remains the responsibility of
        :meth:`Spacecraft.asteroid_in_fov_single_epoch`. Payload SNR is
        evaluated here only for spacecraft that pass FOV, Earth/Moon physical
        occultation, and the enabled dynamic Earth--Moon IV-zone test.

        ``payload_snr.enabled`` and ``payload_snr.shadow_mode`` retain the same
        semantics as the initial-detection stage:

        - disabled: geometry alone controls detection;
        - shadow: SNR is calculated and reported, but geometry controls;
        - gated: geometry and the configured SNR threshold must both pass.

        The detectability test is applied at the initial tracklet epoch only;
        accepted spacecraft generate the complete configured tracklet.

        ``moon_position_eme_km`` is mandatory.  The caller must query the
        authoritative physical Moon at this same asteroid/timer JDTDB epoch
        and pass that one common position to the complete formation.
        """

        def mas_to_rad(x_mas):
            return np.deg2rad(
                np.asarray(x_mas, dtype=float)
                / float(configs["MAS_TO_DEGREE"])
            )

        def result_field(result, field_name, count):
            """Return one flattened result field with deterministic length."""
            value = getattr(result, field_name, None)
            if value is None:
                return np.full(count, np.nan, dtype=float)
            values = np.asarray(value, dtype=float)
            if values.ndim == 0:
                return np.full(count, float(values), dtype=float)
            values = values.reshape(-1)
            if values.size == 1 and count != 1:
                return np.full(count, float(values[0]), dtype=float)
            if values.size != count:
                raise RuntimeError(
                    f"SNR result field {field_name!r} has length "
                    f"{values.size}; expected {count}."
                )
            return values

        asteroid_state = np.asarray(asteroid_state, dtype=float).reshape(6)
        epoch = float(epoch)

        payload_snr_cfg = configs.get("payload_snr", {}) or {}
        snr_enabled = bool(payload_snr_cfg.get("enabled", False))
        snr_shadow_mode = bool(payload_snr_cfg.get("shadow_mode", True))
        snr_mode = (
            "disabled"
            if not snr_enabled
            else ("shadow" if snr_shadow_mode else "gated")
        )

        if snr_enabled:
            if snr_evaluator is None:
                raise ValueError(
                    "Formation.detect requires snr_evaluator when "
                    "payload_snr.enabled is true."
                )
            if absolute_magnitude_h is None:
                raise ValueError(
                    "Formation.detect requires absolute_magnitude_h when "
                    "payload_snr.enabled is true."
                )
            absolute_magnitude_h = float(absolute_magnitude_h)
            if not np.isfinite(absolute_magnitude_h):
                raise ValueError("absolute_magnitude_h must be finite.")

        # One explicitly supplied common Moon position is used by all
        # spacecraft for geometry and SNR at this OD opportunity.
        common_moon_position_km = np.asarray(
            moon_position_eme_km,
            dtype=float,
        ).reshape(3)
        if not np.all(np.isfinite(common_moon_position_km)):
            raise ValueError("The common Moon position must contain finite values.")

        # Initial geometry check. Keep the original geometry flags intact and
        # add separate SNR/active-detection diagnostics below.
        detection_results = [
            sc.asteroid_in_fov_single_epoch(
                asteroid_state[:3],
                epoch,
                configs,
                moon_position_eme_km=common_moon_position_km,
            )
            for sc in self.spacecraft
        ]

        for result in detection_results:
            geometry_detected = bool(result.get("detected", False))
            result["geometry_detected"] = geometry_detected
            result["snr_mode"] = snr_mode
            result["snr_evaluated"] = False
            result["snr_pass"] = None
            result["snr"] = np.nan
            result["snr_threshold"] = (
                float(getattr(snr_evaluator, "snr_threshold", np.nan))
                if snr_enabled
                else np.nan
            )
            result["apparent_magnitude"] = np.nan
            result["apparent_angular_speed_arcsec_s"] = np.nan
            result["trail_length_px"] = np.nan
            result["active_detection"] = geometry_detected

        geometry_candidate_indices = [
            index
            for index, result in enumerate(detection_results)
            if bool(result["geometry_detected"])
        ]

        if snr_enabled and geometry_candidate_indices:
            candidate_indices = np.asarray(
                geometry_candidate_indices,
                dtype=int,
            )
            candidate_count = int(candidate_indices.size)

            et = spice.unitim(epoch, "JDTDB", "ET")
            sun_state, _ = spice.spkgeo(10, et, "J2000", 399)
            sun_position_km = np.asarray(sun_state[:3], dtype=float)
            earth_position_km = np.zeros(3, dtype=float)

            observer_states = np.vstack([
                np.asarray(
                    self.spacecraft[index].curr_state_eme,
                    dtype=float,
                ).reshape(6)
                for index in candidate_indices
            ])
            boresights = np.vstack([
                np.asarray(
                    self.spacecraft[index].boresight,
                    dtype=float,
                ).reshape(3)
                for index in candidate_indices
            ])

            snr_evaluation = snr_evaluator.evaluate_batch(
                absolute_magnitude=np.full(
                    candidate_count,
                    absolute_magnitude_h,
                    dtype=float,
                ),
                asteroid_position_km=np.broadcast_to(
                    asteroid_state[:3],
                    (candidate_count, 3),
                ),
                asteroid_velocity_km_s=np.broadcast_to(
                    asteroid_state[3:],
                    (candidate_count, 3),
                ),
                observer_position_km=observer_states[:, :3],
                observer_velocity_km_s=observer_states[:, 3:],
                sun_position_km=np.broadcast_to(
                    sun_position_km,
                    (candidate_count, 3),
                ),
                earth_position_km=np.broadcast_to(
                    earth_position_km,
                    (candidate_count, 3),
                ),
                moon_position_km=np.broadcast_to(
                    common_moon_position_km,
                    (candidate_count, 3),
                ),
                boresight_unit_vector=boresights,
            )

            pass_mask = np.asarray(
                snr_evaluation.pass_mask,
                dtype=bool,
            ).reshape(-1)
            if pass_mask.size == 1 and candidate_count != 1:
                pass_mask = np.full(
                    candidate_count,
                    bool(pass_mask[0]),
                    dtype=bool,
                )
            if pass_mask.size != candidate_count:
                raise RuntimeError(
                    "SNR pass-mask length does not match the number of "
                    f"geometry candidates: {pass_mask.size} versus "
                    f"{candidate_count}."
                )

            snr_values = result_field(
                snr_evaluation.result,
                "snr",
                candidate_count,
            )
            apparent_magnitude = result_field(
                snr_evaluation.result,
                "apparent_magnitude",
                candidate_count,
            )
            apparent_speed = result_field(
                snr_evaluation.result,
                "apparent_angular_speed_arcsec_s",
                candidate_count,
            )
            trail_length_px = result_field(
                snr_evaluation.result,
                "trail_length_px",
                candidate_count,
            )

            for local_index, spacecraft_index in enumerate(candidate_indices):
                result = detection_results[int(spacecraft_index)]
                result["snr_evaluated"] = True
                result["snr_pass"] = bool(pass_mask[local_index])
                result["snr"] = float(snr_values[local_index])
                result["apparent_magnitude"] = float(
                    apparent_magnitude[local_index]
                )
                result["apparent_angular_speed_arcsec_s"] = float(
                    apparent_speed[local_index]
                )
                result["trail_length_px"] = float(
                    trail_length_px[local_index]
                )

        # Promote SNR into the active detection decision only in gated mode.
        for result in detection_results:
            geometry_detected = bool(result["geometry_detected"])
            if snr_enabled and not snr_shadow_mode:
                active_detection = bool(
                    geometry_detected and result.get("snr_pass", False)
                )
            else:
                active_detection = geometry_detected
            result["active_detection"] = active_detection
            result["detected"] = active_detection

        M = len(self.spacecraft)

        sc_initial_states = np.zeros((M, 6), dtype=float)
        for i, sc in enumerate(self.spacecraft):
            sc_initial_states[i, :] = np.asarray(
                sc.curr_state_eme,
                dtype=float,
            ).reshape(6)

        num_frames = int(configs["number_of_frames"])
        step_days = (
            float(configs["time_between_frames"])
            / float(configs["SECONDS_PER_DAY"])
        )
        epochs = epoch + step_days * np.arange(num_frames, dtype=float)

        ast_states = n_body_prop.propagate(asteroid_state, epoch, epochs)
        sc_states_time_major = n_body_prop.propagate_multiple_objects(
            sc_initial_states,
            epoch,
            epochs,
        )
        sc_states = np.transpose(sc_states_time_major, (1, 0, 2))

        N = num_frames
        perfect_meas = np.full((M, N, 2), np.nan, dtype=float)
        noisy_meas = np.full((M, N, 2), np.nan, dtype=float)

        sigma_ra_rad = mas_to_rad(configs.get("sigma_ra", 0.0))
        sigma_dec_rad = mas_to_rad(configs.get("sigma_dec", 0.0))
        sigma_pointing_rad = mas_to_rad(
            configs.get("sigma_pointing", 0.0)
        )
        sigma_ra_eff = np.sqrt(sigma_ra_rad**2 + sigma_pointing_rad**2)
        sigma_dec_eff = np.sqrt(sigma_dec_rad**2 + sigma_pointing_rad**2)

        eps = 1.0e-12
        for i in range(M):
            if not bool(detection_results[i].get("detected", False)):
                continue

            x_rel = ast_states[:, 0] - sc_states[i, :, 0]
            y_rel = ast_states[:, 1] - sc_states[i, :, 1]
            z_rel = ast_states[:, 2] - sc_states[i, :, 2]

            r_xy = np.hypot(x_rel, y_rel)
            r = np.sqrt(r_xy**2 + z_rel**2)

            ra = np.arctan2(y_rel, x_rel)
            dec = np.arcsin(
                np.clip(
                    z_rel / np.maximum(r, eps),
                    -1.0,
                    1.0,
                )
            )

            perfect_meas[i, :, 0] = ra
            perfect_meas[i, :, 1] = dec

            ra_noisy = ra + np.random.normal(
                loc=0.0,
                scale=sigma_ra_eff,
                size=N,
            )
            dec_noisy = dec + np.random.normal(
                loc=0.0,
                scale=sigma_dec_eff,
                size=N,
            )
            ra_noisy = np.arctan2(np.sin(ra_noisy), np.cos(ra_noisy))

            noisy_meas[i, :, 0] = ra_noisy
            noisy_meas[i, :, 1] = dec_noisy

        return (
            perfect_meas,
            noisy_meas,
            sc_states,
            ast_states,
            epochs,
            detection_results,
        )
