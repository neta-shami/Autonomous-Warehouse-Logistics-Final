from math import hypot
from typing import Optional

from entities.absolute_pose_fix import AbsolutePoseFix
from entities.drive_command import DriveCommand
from entities.enums import LocalizationStatus, PoseSource
from entities.localization_assessment import (
    ConfirmationUpdate,
    LocalizationAssessment,
)
from entities.pose_estimate import PoseEstimate
from entities.sensor_frame import SensorFrame
from entities.value_validation import require_finite, require_nonnegative_integer
from entities.warehouse_topology import WarehouseTopology
from interfaces.i_absolute_localizer import IAbsoluteLocalizer
from interfaces.i_motion_prediction_model import IMotionPredictionModel


class PoseFusionService:
    """Assess one robot's next pose without mutating its accepted estimate."""

    def __init__(
        self,
        motion_model: IMotionPredictionModel,
        localizer: IAbsoluteLocalizer,
        topology: WarehouseTopology,
        *,
        confirmation_samples: int = 3,
        confirmation_tolerance_m: float = 0.1,
        max_speed_mps: float = 1.0,
        max_acceleration_mps2: float = 1.0,
        motion_model_error_m: float = 0.02,
        prediction_error_per_m: float = 0.1,
        prediction_error_per_s: float = 0.05,
        max_fix_age_s: float = 0.5,
        max_prediction_uncertainty_m: float = 0.3,
    ):
        require_nonnegative_integer("confirmation_samples", confirmation_samples)
        if confirmation_samples == 0:
            raise ValueError("confirmation_samples must be at least one")
        for name, value in (
            ("confirmation_tolerance_m", confirmation_tolerance_m),
            ("max_speed_mps", max_speed_mps),
            ("max_acceleration_mps2", max_acceleration_mps2),
            ("motion_model_error_m", motion_model_error_m),
            ("prediction_error_per_m", prediction_error_per_m),
            ("prediction_error_per_s", prediction_error_per_s),
            ("max_fix_age_s", max_fix_age_s),
            ("max_prediction_uncertainty_m", max_prediction_uncertainty_m),
        ):
            require_finite(name, value, minimum=0.0)

        self.motion_model = motion_model
        self.localizer = localizer
        self.topology = topology
        self.confirmation_samples = confirmation_samples
        self.confirmation_tolerance_m = confirmation_tolerance_m
        self.max_speed_mps = max_speed_mps
        self.max_acceleration_mps2 = max_acceleration_mps2
        self.motion_model_error_m = motion_model_error_m
        self.prediction_error_per_m = prediction_error_per_m
        self.prediction_error_per_s = prediction_error_per_s
        self.max_fix_age_s = max_fix_age_s
        self.max_prediction_uncertainty_m = max_prediction_uncertainty_m
        self._candidate: Optional[AbsolutePoseFix] = None
        self._consecutive_samples = 0

    def assess(
        self,
        frame: Optional[SensorFrame],
        previous_estimate: Optional[PoseEstimate],
        previous_status: LocalizationStatus,
        last_applied_command: DriveCommand,
        dt: float,
    ) -> LocalizationAssessment:
        require_finite("fusion dt", dt, minimum=0.0)
        if not isinstance(previous_status, LocalizationStatus):
            raise TypeError("previous_status must be LocalizationStatus")
        if not isinstance(last_applied_command, DriveCommand):
            raise TypeError("last_applied_command must be DriveCommand")
        if previous_estimate is not None and not isinstance(
            previous_estimate, PoseEstimate
        ):
            raise TypeError("previous_estimate must be PoseEstimate or None")

        fix = None if frame is None else self.localizer.locate(frame, self.topology)
        if fix is not None and frame is not None:
            return self._assess_fix(
                fix, previous_estimate, dt, frame.robot_id
            )
        return self._assess_without_fix(
            previous_estimate,
            previous_status,
            last_applied_command,
            dt,
        )

    def commit_confirmation(self, update: ConfirmationUpdate) -> None:
        """Commit only the candidate state proposed by an accepted assessment."""
        if not isinstance(update, ConfirmationUpdate):
            raise TypeError("update must be ConfirmationUpdate")
        self._candidate = update.candidate
        self._consecutive_samples = update.consecutive_samples

    def _assess_fix(
        self,
        fix: AbsolutePoseFix,
        previous: Optional[PoseEstimate],
        dt: float,
        robot_id: str,
    ) -> LocalizationAssessment:
        if previous is None:
            confirmation = self._propose_confirmation(fix)
            if confirmation.confirmed:
                return self._trusted_absolute(fix, confirmation)
            return self._stopped(
                LocalizationStatus.UNINITIALIZED,
                confirmation=confirmation,
            )

        divergence = previous.pose.distance_to(fix.pose)
        physical_motion_bound = (
            self.max_speed_mps * dt
            + 0.5 * self.max_acceleration_mps2 * dt * dt
            + self.motion_model_error_m
        )
        reachable_bound = (
            physical_motion_bound
            + previous.uncertainty_m
            + fix.error_bound_m
        )
        if divergence <= reachable_bound:
            estimate = PoseEstimate(
                fix.pose,
                fix.error_bound_m,
                PoseSource.FUSED,
                fix.sim_time,
            )
            return LocalizationAssessment(
                estimate,
                LocalizationStatus.TRUSTED,
                False,
                None,
                self._reset_confirmation(),
                (),
            )

        confirmation = self._propose_confirmation(fix)
        if confirmation.confirmed:
            trusted = self._trusted_absolute(fix, confirmation)
            return LocalizationAssessment(
                trusted.estimate,
                trusted.status,
                trusted.stop_required,
                divergence,
                confirmation,
                (),
            )
        return self._stopped(
            LocalizationStatus.LOST,
            divergence=divergence,
            confirmation=confirmation,
            events=(),
        )

    def _assess_without_fix(
        self,
        previous: Optional[PoseEstimate],
        previous_status: LocalizationStatus,
        command: DriveCommand,
        dt: float,
    ) -> LocalizationAssessment:
        reset = self._reset_confirmation()
        if (
            previous is None
            or previous_status
            not in (LocalizationStatus.TRUSTED, LocalizationStatus.DEGRADED)
            or dt > self.max_fix_age_s
        ):
            status = (
                LocalizationStatus.UNINITIALIZED
                if previous is None
                else LocalizationStatus.LOST
            )
            return self._stopped(status, confirmation=reset)

        predicted_pose = self.motion_model.predict(previous.pose, command, dt)
        commanded_distance = hypot(command.vx, command.vy) * dt
        uncertainty = (
            previous.uncertainty_m
            + commanded_distance * self.prediction_error_per_m
            + dt * self.prediction_error_per_s
        )
        if uncertainty > self.max_prediction_uncertainty_m:
            return self._stopped(LocalizationStatus.LOST, confirmation=reset)

        estimate = PoseEstimate(
            predicted_pose,
            uncertainty,
            PoseSource.PREDICTED,
            previous.sim_time + dt,
        )
        return LocalizationAssessment(
            estimate,
            LocalizationStatus.DEGRADED,
            False,
            None,
            reset,
            (),
        )

    def _propose_confirmation(self, fix: AbsolutePoseFix) -> ConfirmationUpdate:
        if self._candidate is None:
            count = 1
        else:
            consistency_bound = (
                self.confirmation_tolerance_m
                + self._candidate.error_bound_m
                + fix.error_bound_m
            )
            count = (
                self._consecutive_samples + 1
                if self._candidate.pose.distance_to(fix.pose) <= consistency_bound
                else 1
            )
        return ConfirmationUpdate(
            fix,
            count,
            count >= self.confirmation_samples,
        )

    @staticmethod
    def _reset_confirmation() -> ConfirmationUpdate:
        return ConfirmationUpdate(None, 0, False)

    @staticmethod
    def _trusted_absolute(
        fix: AbsolutePoseFix, confirmation: ConfirmationUpdate
    ) -> LocalizationAssessment:
        estimate = PoseEstimate(
            fix.pose,
            fix.error_bound_m,
            PoseSource.ABSOLUTE_FIX,
            fix.sim_time,
        )
        return LocalizationAssessment(
            estimate,
            LocalizationStatus.TRUSTED,
            False,
            None,
            confirmation,
            (),
        )

    @staticmethod
    def _stopped(
        status: LocalizationStatus,
        *,
        divergence: Optional[float] = None,
        confirmation: ConfirmationUpdate,
        events=(),
    ) -> LocalizationAssessment:
        return LocalizationAssessment(
            None,
            status,
            True,
            divergence,
            confirmation,
            events,
        )
