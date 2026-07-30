import math

import numpy as np


class SimTime:
    """Track the simulated mission timeline for IOD/OD follow-up cycles.

    ``curr_epoch`` remains the start epoch of the current observation cycle.
    A cycle advances only when ``step`` is called with the selected attitude-
    coordination completion epoch (or with a prediction-only/termination epoch).
    """

    def __init__(
        self,
        configs,
        current_od_index=None,
        current_epoch=None,
        iod_time=None,
        current_integration_epoch=None,
        current_integration_index=None,
        attitude_coordination_expected_time=None,
    ):
        self.curr_od_index = current_od_index
        self.curr_epoch = current_epoch  # JDTDB; start of the current cycle
        self.curr_integration_epoch = current_integration_epoch
        self.curr_integration_index = current_integration_index
        self.seconds_per_day = float(configs.get("SECONDS_PER_DAY", 86400.0))
        self.end_time = self.curr_epoch + float(configs["od_duration_days"])

        # Candidate slew-duration grid searched by attitude coordination.
        epochs = configs.get("epochs", {}) or {}
        self.attcoord_dtmin = float(epochs.get("dt_min", 10.0))
        self.attcoord_dtmax = float(epochs.get("dt_max", 600.0))
        self.attcoord_numdt = int(epochs.get("n_dt", 60))
        if self.attcoord_numdt <= 0:
            raise ValueError("epochs.n_dt must be positive.")
        if self.attcoord_dtmin < 0.0 or self.attcoord_dtmax < self.attcoord_dtmin:
            raise ValueError(
                "epochs.dt_min and epochs.dt_max must satisfy "
                "0 <= dt_min <= dt_max."
            )

        # Keep the full grid immutable. The active grid is rebuilt and trimmed
        # independently on every cycle, preserving end-of-run protection.
        self.attcoord_searchtimes_base = np.linspace(
            self.attcoord_dtmin,
            self.attcoord_dtmax,
            self.attcoord_numdt,
            dtype=float,
        )
        self.attcoord_searchtimes = self.attcoord_searchtimes_base.copy()
        self.attcoord_searchtimes_jdtdb = None

        self.attcoord_time = None  # measured Python runtime; diagnostics only
        if attitude_coordination_expected_time is None:
            average_epoch_time = float(epochs.get("average_epoch_time", 0.0))
            self.attcoord_expectedtime = (
                len(self.attcoord_searchtimes_base) * average_epoch_time
            )
        else:
            self.attcoord_expectedtime = float(
                attitude_coordination_expected_time
            )

        # Tracklet and onboard-detection delays.
        self.datacollect_time = (
            int(configs["number_of_frames"])
            * float(configs["time_between_frames"])
        )
        self.detection_patchtime = float(configs["patch_time"])
        big_h, big_l = (
            int(configs["number_of_pixels"][0]),
            int(configs["number_of_pixels"][1]),
        )
        patch_h, patch_l = (
            int(configs["patch_size"][0]),
            int(configs["patch_size"][1]),
        )
        if patch_h <= 0 or patch_l <= 0:
            raise ValueError("patch_size entries must be positive.")
        self.detection_time = (
            int(math.ceil((big_h * big_l) / (patch_h * patch_l))) + 1
        ) * self.detection_patchtime

        # Additional operational delays. Defaults preserve legacy behavior.
        timing = configs.get("timing", {}) or {}
        self.preprocessing_time = self._nonnegative_seconds(
            timing.get("preprocessing_time_sec", 0.0),
            "timing.preprocessing_time_sec",
        )
        self.measurement_comm_time = self._nonnegative_seconds(
            timing.get("measurement_comm_time_sec", 0.0),
            "timing.measurement_comm_time_sec",
        )
        self.boresight_comm_time = self._nonnegative_seconds(
            timing.get("boresight_comm_time_sec", 0.0),
            "timing.boresight_comm_time_sec",
        )

        self.iod_time = None if iod_time is None else float(iod_time)
        self.slew_time = None  # selected slew duration [s]
        self.od_time = None  # last measured OD computation duration [s]

        # Most recently constructed timing budget, useful for diagnostics and
        # automatically preserved by existing checkpoint serialization.
        self.last_timing_breakdown = {}

    @staticmethod
    def _nonnegative_seconds(value, name):
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return value

    def _navigation_time_seconds(self, od_time=None):
        """Return the applicable IOD/OD computation delay for this cycle."""
        if od_time is not None:
            return self._nonnegative_seconds(od_time, "od_time")

        if int(self.curr_od_index or 0) == 0:
            return 0.0 if self.iod_time is None else self._nonnegative_seconds(
                self.iod_time, "iod_time"
            )

        return 0.0 if self.od_time is None else self._nonnegative_seconds(
            self.od_time, "self.od_time"
        )

    def _pipeline_delay_seconds(self, od_time=None, measurements_available=True):
        """Return fixed delay before the candidate slew-duration grid.

        With measurements available, the sequence is:

          collect -> preprocess -> detect -> measurement crosslink -> IOD/OD
          -> attitude coordination -> boresight-command crosslink.

        For a no-detection reacquisition attempt, there are no measurements to
        transmit and no measurement update to compute, so the middle two terms
        are omitted while collection, detection, AC, command transmission, and
        candidate slew timing remain.
        """
        measurements_available = bool(measurements_available)
        navigation_time = (
            self._navigation_time_seconds(od_time)
            if measurements_available
            else 0.0
        )
        measurement_comm_time = (
            self.measurement_comm_time if measurements_available else 0.0
        )

        breakdown = {
            "data_collection_time_sec": float(self.datacollect_time),
            "preprocessing_time_sec": float(self.preprocessing_time),
            "detection_time_sec": float(self.detection_time),
            "measurement_comm_time_sec": float(measurement_comm_time),
            "navigation_time_sec": float(navigation_time),
            "attcoord_expected_time_sec": float(self.attcoord_expectedtime),
            "boresight_comm_time_sec": float(self.boresight_comm_time),
            "measurements_available": measurements_available,
        }
        breakdown["fixed_pre_slew_time_sec"] = float(
            sum(
                breakdown[key]
                for key in (
                    "data_collection_time_sec",
                    "preprocessing_time_sec",
                    "detection_time_sec",
                    "measurement_comm_time_sec",
                    "navigation_time_sec",
                    "attcoord_expected_time_sec",
                    "boresight_comm_time_sec",
                )
            )
        )
        return breakdown

    def set_attcoord_searchtimes(self, od_time=None, measurements_available=True):
        """Build absolute candidate epochs and trim them against ``end_time``.

        The candidate values themselves remain pure slew durations. Every call
        starts from ``attcoord_searchtimes_base``, so a prior truncation cannot
        permanently remove candidates from later cycles.
        """
        measurements_available = bool(measurements_available)
        if od_time is not None and measurements_available:
            # Retain the most recent real OD-update runtime as the estimate used
            # by the next cycle's pre-check. Prediction-only no-detection work
            # does not replace that value.
            self.od_time = self._nonnegative_seconds(od_time, "od_time")

        breakdown = self._pipeline_delay_seconds(
            od_time=od_time,
            measurements_available=measurements_available,
        )
        fixed_delay_sec = breakdown["fixed_pre_slew_time_sec"]

        base_grid = np.asarray(
            self.attcoord_searchtimes_base, dtype=float
        ).reshape(-1)
        candidate_epochs = (
            float(self.curr_epoch)
            + (fixed_delay_sec + base_grid) / self.seconds_per_day
        )
        valid = candidate_epochs <= float(self.end_time) + 1.0e-15

        self.attcoord_searchtimes = base_grid[valid].copy()
        self.attcoord_searchtimes_jdtdb = candidate_epochs[valid].copy()
        self.last_timing_breakdown = breakdown
        return

    def get_no_detection_decision_epoch(self):
        """Epoch when an ordinary no-detection result becomes available.

        This includes acquisition of the complete tracklet, preprocessing, and
        execution of the detection algorithm, but no measurement crosslink,
        IOD/OD update, attitude coordination, command transmission, or slew.
        """
        delay_sec = (
            float(self.datacollect_time)
            + float(self.preprocessing_time)
            + float(self.detection_time)
        )
        return float(self.curr_epoch) + delay_sec / self.seconds_per_day

    def set_attcoord_time(self, time):
        self.attcoord_time = float(time)
        return

    def set_slew_time(self, time):
        self.slew_time = float(time)

    def step(self, epoch, best_anchor_index, best_anchor_epoch):
        self.curr_epoch = float(epoch)
        self.curr_od_index += 1
        self.curr_integration_epoch = best_anchor_epoch
        self.curr_integration_index = best_anchor_index

    def check_search_times(self, measurements_available=True):
        """Return True when no complete candidate epoch fits before end time.

        This is a non-mutating pre-check. It uses the initial IOD delay for the
        first cycle and the most recent measured OD-update delay thereafter.
        """
        breakdown = self._pipeline_delay_seconds(
            od_time=None,
            measurements_available=measurements_available,
        )
        fixed_delay_sec = breakdown["fixed_pre_slew_time_sec"]
        base_grid = np.asarray(
            self.attcoord_searchtimes_base, dtype=float
        ).reshape(-1)
        candidate_epochs = (
            float(self.curr_epoch)
            + (fixed_delay_sec + base_grid) / self.seconds_per_day
        )
        return not bool(
            np.any(candidate_epochs <= float(self.end_time) + 1.0e-15)
        )
