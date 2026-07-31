
import numpy as np
from earth_moon_invisibility_zone import (
    compute_earth_moon_invisibility_zone_batch,
    line_of_sight_inside_invisibility_zone,
)


class Spacecraft:


    def __init__(self, ini_pos, ini_pos_index, configs, current_state_eme=None, current_spacecraftepoch=None, current_boresight=None):
        self.ini_position = ini_pos  # initial position of the spacecraft in the quasi-halo orbit
        self.ini_pos_index = ini_pos_index  # initial position index in the quasi-halo orbit csv
        self.velocity = None
        self.position = None
        self.boresight = np.array([-1, 0, 0]) if current_boresight is None else current_boresight
        self.pixel_scale = float(
            configs['payload_snr']['optics']['pixel_scale_arcsec_per_px']
        )
        self.fov = configs['fov']
        self.number_of_pixels = configs['number_of_pixels']
        self.reaction_wheel_torque = configs['reaction_wheel_torque']
        self.reaction_wheel_momentum = configs['reaction_wheel_momentum']
        self.mass = configs['mass']
        self.length = configs['length']
        self.telescope_diameter = configs['telescope_diameter']
        self.telescope_length = configs['telescope_length']
        self.telescope_mass = configs['telescope_mass']
        self.telescope_offset = configs['telescope_offset']
        self.sigma_ra = configs['sigma_ra']
        self.sigma_dec = configs['sigma_dec']
        self.sigma_pointing = configs['sigma_pointing']
        self.matched_trajectory = None  # this contains an array of the trajectory of the sc that has same length as the asteroid traj in question
        self.matched_trajectory_full = None
        self.curr_state_eme = current_state_eme
        self.curr_sc_epoch = current_spacecraftepoch
        return


    def set_state(self, position, velocity):
        self.position = position
        self.velocity = velocity
        return

    def get_spacecraft_pos(self, index):
        return self.matched_trajectory[index, :]


    def get_attitude(self):
        raise NotImplementedError


    def is_occluded_batch(self, spacecraft_pos, asteroid_pos, earth_pos, moon_pos, configs):
        """Vectorized occlusion check for all time steps."""
        def check_occlusion(body_pos, body_radius):
            sc_to_ast = asteroid_pos - spacecraft_pos  # (N,3)
            sc_to_body = body_pos - spacecraft_pos  # (N,3)

            proj_length = np.einsum('ij,ij->i', sc_to_body, sc_to_ast) / np.linalg.norm(sc_to_ast, axis=1)
            proj_point = spacecraft_pos + (sc_to_ast / np.linalg.norm(sc_to_ast, axis=1)[:, None]) * proj_length[:, None]

            min_distance = np.linalg.norm(proj_point - body_pos, axis=1)
            occluded = (min_distance < body_radius) & (proj_length > 0) & (proj_length < np.linalg.norm(sc_to_ast, axis=1))
            return occluded

        return (check_occlusion(earth_pos, configs['EARTH_RADIUS_KM'] * configs['KM_TO_M'] / configs['AU_TO_M'])
                | check_occlusion(moon_pos, configs['MOON_RADIUS_KM'] * configs['KM_TO_M']  / configs['AU_TO_M']))

    def asteroid_in_fov_batch_old(self,
                              asteroid_trajectory,  # (N,3) in same frame as s/c & boresight
                              spacecraft_position,  # (N,3)
                              earth_position,  # (N,3)
                              moon_position,  # (N,3)
                              configs):
        """
        Determine when the asteroid is inside the spacecraft's conical FOV,
        accounting for Earth/Moon occlusion.

        Returns:
            result (N,) array:
                index where visible, -1 where not visible
        """

        positions = np.asarray(asteroid_trajectory, dtype=float)
        sc_pos = np.asarray(spacecraft_position, dtype=float)

        # ---- Correct half-angle from FOV area ----
        def fov_deg2_to_half_angle_rad(FOV_deg2):
            """
            Convert sky area FOV (deg^2) to cone half-angle (radians)
            using spherical cap geometry.
            """
            return np.arccos(
                1.0 - (FOV_deg2 / (180.0 / np.pi) ** 2) / (2.0 * np.pi)
            )
        theta_h = fov_deg2_to_half_angle_rad(self.fov)
        cos_theta_h = np.cos(theta_h)

        # ---- Normalize boresight once ----
        b = np.asarray(self.boresight, dtype=float)
        b = b / (np.linalg.norm(b) + 1e-15)

        # ---- Relative vectors from s/c to asteroid ----
        rel_pos = positions - sc_pos  # (N,3)
        rel_norm = np.linalg.norm(rel_pos, axis=1)  # (N,)

        # ---- Use cosine test instead of angle ----
        vhat = rel_pos / (rel_norm[:, None] + 1e-15)
        cos_angles = vhat @ b  # (N,)

        # Inside conical FOV
        in_fov = cos_angles >= cos_theta_h

        # ---- Occlusion check (unchanged) ----
        occluded = self.is_occluded_batch(
            spacecraft_position,
            positions,
            earth_position,
            moon_position,
            configs
        )

        visible = in_fov & (~occluded)

        # ---- Output format exactly like your original ----
        result = np.full(positions.shape[0], -1.0, dtype=float)
        visible_indices = np.where(visible)[0]
        result[visible_indices] = visible_indices

        return result


    def asteroid_in_fov_batch(self, asteroid_trajectory, spacecraft_position,
                              earth_position, moon_position, configs):
        """Evaluate FOV, physical occultation, and dynamic EMS exclusion.

        All Cartesian inputs use the same frame and distance unit. The dynamic
        Earth--Moon invisibility zone is a fixed-half-angle cone whose axis is
        the instantaneous angular bisector of the spacecraft-to-Earth and
        spacecraft-to-Moon line-of-sight directions.

        Returns
        -------
        result_base : ndarray, shape (N,)
            Index where FOV and Earth/Moon occultation pass, otherwise -1.
        result_ems_filtered : ndarray, shape (N,)
            Same result after excluding target LOS directions inside the
            dynamic Earth--Moon invisibility-zone cone.
        """
        positions = np.asarray(asteroid_trajectory, dtype=float)
        sc_pos = np.asarray(spacecraft_position, dtype=float)
        earth_pos = np.asarray(earth_position, dtype=float)
        moon_pos = np.asarray(moon_position, dtype=float)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"asteroid_trajectory must have shape (N,3), got {positions.shape}."
            )
        if sc_pos.shape != positions.shape:
            raise ValueError(
                "spacecraft_position must have the same shape as "
                f"asteroid_trajectory, got {sc_pos.shape} and {positions.shape}."
            )

        n_epochs = positions.shape[0]

        def fov_deg2_to_half_angle_rad(fov_deg2):
            return np.arccos(
                1.0 - (fov_deg2 / (180.0 / np.pi) ** 2) / (2.0 * np.pi)
            )

        theta_h = fov_deg2_to_half_angle_rad(self.fov)
        cos_theta_h = np.cos(theta_h)

        b = np.asarray(self.boresight, dtype=float).reshape(3)
        b_norm = np.linalg.norm(b)
        if b_norm <= 1.0e-15:
            raise ValueError("self.boresight has zero norm.")
        b_hat = b / b_norm

        rel_pos = positions - sc_pos
        rel_norm = np.linalg.norm(rel_pos, axis=1)
        valid_rel = rel_norm > 1.0e-15
        target_los = np.zeros_like(rel_pos)
        target_los[valid_rel] = rel_pos[valid_rel] / rel_norm[valid_rel, None]

        cos_angles = np.full(n_epochs, -np.inf, dtype=float)
        cos_angles[valid_rel] = target_los[valid_rel] @ b_hat
        in_fov = valid_rel & (cos_angles >= cos_theta_h)

        occluded_em = self.is_occluded_batch(
            sc_pos,
            positions,
            earth_pos,
            moon_pos,
            configs,
        )
        visible_base = in_fov & (~occluded_em)

        result_base = np.full(n_epochs, -1.0, dtype=float)
        base_idx = np.flatnonzero(visible_base)
        result_base[base_idx] = base_idx

        ems_conf = configs.get("ems", {}) or {}
        ems_enabled = bool(ems_conf.get("enabled", False))
        if not ems_enabled:
            return result_base, result_base.copy()

        half_angle_deg = float(ems_conf["half_angle_deg"])
        zone = compute_earth_moon_invisibility_zone_batch(
            spacecraft_position=sc_pos,
            earth_position=earth_pos,
            moon_position=moon_pos,
            half_angle_deg=half_angle_deg,
        )
        inside_ems, _ = line_of_sight_inside_invisibility_zone(
            target_position=positions,
            spacecraft_position=sc_pos,
            zone=zone,
        )
        inside_ems = np.asarray(inside_ems, dtype=bool)

        visible_ems_filtered = visible_base & (~inside_ems)
        result_ems_filtered = np.full(n_epochs, -1.0, dtype=float)
        ems_idx = np.flatnonzero(visible_ems_filtered)
        result_ems_filtered[ems_idx] = ems_idx

        return result_base, result_ems_filtered

    def asteroid_in_fov_single_epoch(
            self,
            asteroid_position_km,
            jdtdb,
            configs,
            *,
            moon_position_eme_km,
    ):
        """Single-epoch detectability in geocentric EME/J2000 coordinates.

        Detection requires the asteroid to be inside the payload FOV, not
        physically occulted by Earth or the Moon, and outside the dynamic
        Earth--Moon invisibility-zone cone when that cone is enabled.

        ``moon_position_eme_km`` is required and must be the one common
        physical Moon position selected by the caller for this simulation
        epoch.  This method never chooses or interpolates an ephemeris source.
        """

        def fov_deg2_to_half_angle_rad(fov_deg2):
            return np.arccos(
                1.0 - (fov_deg2 / (180.0 / np.pi) ** 2) / (2.0 * np.pi)
            )

        def check_occlusion_single(
            spacecraft_pos, asteroid_pos, body_pos, body_radius
        ):
            sc_to_ast = asteroid_pos - spacecraft_pos
            sc_to_body = body_pos - spacecraft_pos
            sc_to_ast_norm = np.linalg.norm(sc_to_ast)
            if sc_to_ast_norm <= 1.0e-15:
                return False

            proj_length = np.dot(sc_to_body, sc_to_ast) / sc_to_ast_norm
            proj_point = (
                spacecraft_pos
                + (sc_to_ast / sc_to_ast_norm) * proj_length
            )
            min_distance = np.linalg.norm(proj_point - body_pos)
            return bool(
                (min_distance < body_radius)
                and (proj_length > 0.0)
                and (proj_length < sc_to_ast_norm)
            )

        asteroid_position_km = np.asarray(
            asteroid_position_km, dtype=float
        ).reshape(3)
        if self.curr_state_eme is None:
            raise ValueError(
                "curr_state_eme is required for a single-epoch visibility check."
            )
        sc_pos_km = np.asarray(self.curr_state_eme[:3], dtype=float).reshape(3)

        earth_pos_km = np.zeros(3, dtype=float)
        moon_pos_km = np.asarray(
            moon_position_eme_km,
            dtype=float,
        ).reshape(3)
        if not np.all(np.isfinite(moon_pos_km)):
            raise ValueError(
                "moon_position_eme_km must contain finite values."
            )

        earth_radius_km = float(configs["EARTH_RADIUS_KM"])
        moon_radius_km = float(configs["MOON_RADIUS_KM"])

        theta_h = fov_deg2_to_half_angle_rad(self.fov)
        cos_theta_h = np.cos(theta_h)

        b = np.asarray(self.boresight, dtype=float).reshape(3)
        b_norm = np.linalg.norm(b)
        if b_norm <= 1.0e-15:
            raise ValueError("self.boresight has zero norm.")
        b_hat = b / b_norm

        rel_pos = asteroid_position_km - sc_pos_km
        rel_norm = np.linalg.norm(rel_pos)
        if rel_norm <= 1.0e-15:
            return {
                "detected": False,
                "in_fov": False,
                "occluded_earth": False,
                "occluded_moon": False,
                "occluded_em": False,
                "occluded_ems": False,
                "visible_base": False,
                "visible_ems_filtered": False,
                "cos_angle": None,
                "cos_theta_h": float(cos_theta_h),
                "iv_axis_geo_eme": None,
                "target_iv_axis_separation_rad": None,
                "iv_half_angle_rad": None,
                "earth_moon_angular_separation_rad": None,
            }

        target_los = rel_pos / rel_norm
        cos_angle = float(np.dot(target_los, b_hat))
        in_fov = bool(cos_angle >= cos_theta_h)

        occluded_earth = check_occlusion_single(
            sc_pos_km,
            asteroid_position_km,
            earth_pos_km,
            earth_radius_km,
        )
        occluded_moon = check_occlusion_single(
            sc_pos_km,
            asteroid_position_km,
            moon_pos_km,
            moon_radius_km,
        )
        occluded_em = bool(occluded_earth or occluded_moon)
        visible_base = bool(in_fov and not occluded_em)

        ems_conf = configs.get("ems", {}) or {}
        ems_enabled = bool(ems_conf.get("enabled", False))

        occluded_ems = False
        iv_axis_geo_eme = None
        target_iv_axis_separation_rad = None
        iv_half_angle_rad = None
        earth_moon_angular_separation_rad = None

        if ems_enabled:
            zone = compute_earth_moon_invisibility_zone_batch(
                spacecraft_position=sc_pos_km,
                earth_position=earth_pos_km,
                moon_position=moon_pos_km,
                half_angle_deg=float(ems_conf["half_angle_deg"]),
            )
            inside_ems, target_separation = (
                line_of_sight_inside_invisibility_zone(
                    target_position=asteroid_position_km,
                    spacecraft_position=sc_pos_km,
                    zone=zone,
                )
            )
            occluded_ems = bool(inside_ems)
            iv_axis_geo_eme = np.asarray(zone.axis_geo_eme, dtype=float)
            target_iv_axis_separation_rad = float(target_separation)
            iv_half_angle_rad = float(zone.half_angle_rad)
            earth_moon_angular_separation_rad = float(
                zone.earth_moon_angular_separation_rad
            )

        visible_ems_filtered = bool(visible_base and not occluded_ems)

        return {
            "detected": visible_ems_filtered,
            "in_fov": in_fov,
            "occluded_earth": occluded_earth,
            "occluded_moon": occluded_moon,
            "occluded_em": occluded_em,
            "occluded_ems": occluded_ems,
            "visible_base": visible_base,
            "visible_ems_filtered": visible_ems_filtered,
            "cos_angle": cos_angle,
            "cos_theta_h": float(cos_theta_h),
            "iv_axis_geo_eme": iv_axis_geo_eme,
            "target_iv_axis_separation_rad": target_iv_axis_separation_rad,
            "iv_half_angle_rad": iv_half_angle_rad,
            "earth_moon_angular_separation_rad": (
                earth_moon_angular_separation_rad
            ),
        }

    def asteroid_in_fov_batch_km_geocentric(
            self,
            asteroid_trajectory_km,
            spacecraft_trajectory_km,
            spacecraft_boresight_eme,
            jdtdb_list,
            configs,
            *,
            moon_positions_eme_km,
    ):
        """Evaluate cumulative detection masks in geocentric EME/J2000.

        The masks are cumulative:

        1. ``mask_fov``: target is inside the conical payload FOV.
        2. ``mask_fov_occultation``: FOV passes and Earth/Moon do not
           physically occult the spacecraft-to-target line of sight.
        3. ``mask_fov_occultation_ems``: the first two tests pass and the
           target LOS lies outside the dynamic Earth--Moon invisibility-zone
           cone.

        The dynamic cone has a fixed half-angle from
        ``configs["ems"]["half_angle_deg"]`` and an axis equal to the
        instantaneous angular bisector of the spacecraft-to-Earth and
        spacecraft-to-Moon LOS directions. If ``configs["ems"]["enabled"]``
        is false, the third mask equals the second mask.

        ``moon_positions_eme_km`` is required.  It must contain the common
        physical Moon history queried at the same ``jdtdb_list`` epochs. This
        method does not interpolate Moon states or select an ephemeris source.
        """

        def fov_deg2_to_half_angle_rad(fov_deg2):
            return np.arccos(
                1.0 - (fov_deg2 / (180.0 / np.pi) ** 2) / (2.0 * np.pi)
            )

        def check_occlusion_batch(
            spacecraft_pos, asteroid_pos, body_pos, body_radius
        ):
            sc_to_ast = asteroid_pos - spacecraft_pos
            sc_to_body = body_pos - spacecraft_pos

            sc_to_ast_norm = np.linalg.norm(sc_to_ast, axis=1)
            valid = sc_to_ast_norm > 1.0e-15

            proj_length = np.zeros_like(sc_to_ast_norm)
            proj_length[valid] = (
                np.einsum("ij,ij->i", sc_to_body[valid], sc_to_ast[valid])
                / sc_to_ast_norm[valid]
            )

            proj_point = spacecraft_pos.copy()
            proj_point[valid] = (
                spacecraft_pos[valid]
                + (sc_to_ast[valid] / sc_to_ast_norm[valid, None])
                * proj_length[valid, None]
            )
            min_distance = np.linalg.norm(proj_point - body_pos, axis=1)

            return (
                valid
                & (min_distance < body_radius)
                & (proj_length > 0.0)
                & (proj_length < sc_to_ast_norm)
            )

        positions = np.asarray(asteroid_trajectory_km, dtype=float)
        sc_pos = np.asarray(spacecraft_trajectory_km, dtype=float)
        boresight = np.asarray(spacecraft_boresight_eme, dtype=float)
        jdtdb = np.asarray(jdtdb_list, dtype=float).reshape(-1)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                "asteroid_trajectory_km must have shape (N,3), got "
                f"{positions.shape}."
            )
        if sc_pos.ndim != 2 or sc_pos.shape[1] != 3:
            raise ValueError(
                "spacecraft_trajectory_km must have shape (N,3), got "
                f"{sc_pos.shape}."
            )

        n_epochs = positions.shape[0]
        if sc_pos.shape[0] != n_epochs or jdtdb.shape[0] != n_epochs:
            raise ValueError(
                f"Inconsistent lengths: asteroid={positions.shape[0]}, "
                f"spacecraft={sc_pos.shape[0]}, jdtdb={jdtdb.shape[0]}."
            )

        if boresight.ndim == 1:
            if boresight.shape != (3,):
                raise ValueError(
                    "spacecraft_boresight_eme must have shape (3,) or "
                    f"(N,3), got {boresight.shape}."
                )
            boresight = np.broadcast_to(
                boresight.reshape(1, 3), (n_epochs, 3)
            ).copy()
        elif boresight.shape != (n_epochs, 3):
            raise ValueError(
                "spacecraft_boresight_eme must have shape (3,) or "
                f"({n_epochs},3), got {boresight.shape}."
            )

        boresight_norm = np.linalg.norm(boresight, axis=1)
        if np.any(boresight_norm <= 1.0e-15):
            raise ValueError(
                "spacecraft_boresight_eme has zero-norm vectors at indices "
                f"{np.flatnonzero(boresight_norm <= 1.0e-15).tolist()}."
            )
        b_hat = boresight / boresight_norm[:, None]

        earth_pos = np.zeros((n_epochs, 3), dtype=float)
        moon_pos = np.asarray(moon_positions_eme_km, dtype=float)
        if moon_pos.shape != (n_epochs, 3):
            raise ValueError(
                "moon_positions_eme_km must have shape "
                f"({n_epochs},3), got {moon_pos.shape}."
            )
        if not np.all(np.isfinite(moon_pos)):
            raise ValueError(
                "moon_positions_eme_km must contain only finite values."
            )

        earth_radius_km = float(configs["EARTH_RADIUS_KM"])
        moon_radius_km = float(configs["MOON_RADIUS_KM"])

        theta_h = fov_deg2_to_half_angle_rad(self.fov)
        cos_theta_h = np.cos(theta_h)

        rel_pos = positions - sc_pos
        rel_norm = np.linalg.norm(rel_pos, axis=1)
        valid_rel = rel_norm > 1.0e-15
        target_los = np.zeros_like(rel_pos)
        target_los[valid_rel] = (
            rel_pos[valid_rel] / rel_norm[valid_rel, None]
        )

        cos_angles = np.full(n_epochs, -np.inf, dtype=float)
        cos_angles[valid_rel] = np.einsum(
            "ij,ij->i", target_los[valid_rel], b_hat[valid_rel]
        )
        mask_fov = valid_rel & (cos_angles >= cos_theta_h)

        occluded_earth = check_occlusion_batch(
            sc_pos, positions, earth_pos, earth_radius_km
        )
        occluded_moon = check_occlusion_batch(
            sc_pos, positions, moon_pos, moon_radius_km
        )
        mask_fov_occultation = (
            mask_fov & (~(occluded_earth | occluded_moon))
        )

        ems_conf = configs.get("ems", {}) or {}
        ems_enabled = bool(ems_conf.get("enabled", False))

        if ems_enabled:
            zone = compute_earth_moon_invisibility_zone_batch(
                spacecraft_position=sc_pos,
                earth_position=earth_pos,
                moon_position=moon_pos,
                half_angle_deg=float(ems_conf["half_angle_deg"]),
            )
            inside_ems, _ = line_of_sight_inside_invisibility_zone(
                target_position=positions,
                spacecraft_position=sc_pos,
                zone=zone,
            )
            inside_ems = np.asarray(inside_ems, dtype=bool)
            mask_fov_occultation_ems = (
                mask_fov_occultation & (~inside_ems)
            )
        else:
            mask_fov_occultation_ems = mask_fov_occultation.copy()

        if np.any(mask_fov_occultation & ~mask_fov):
            raise RuntimeError(
                "mask_fov_occultation contains epochs that failed mask_fov."
            )
        if np.any(mask_fov_occultation_ems & ~mask_fov_occultation):
            raise RuntimeError(
                "mask_fov_occultation_ems contains epochs that failed "
                "mask_fov_occultation."
            )

        return (
            mask_fov,
            mask_fov_occultation,
            mask_fov_occultation_ems,
        )
