"""Pure safety and state-transition tests for pose fusion."""

from unittest.mock import MagicMock

import pytest

from entities.absolute_pose_fix import AbsolutePoseFix
from entities.drive_command import DriveCommand
from entities.enums import LocalizationStatus, PoseSource
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from entities.sensor_frame import SensorFrame
from use_cases.holonomic_command_motion_model import HolonomicCommandMotionModel
from use_cases.pose_fusion_service import PoseFusionService


def _frame(sim_time=1.0, velocity=(0.0, 0.0)):
    return SensorFrame("r1", velocity, (), None, sim_time, 1)


def _fix(x, y, sim_time=1.0, error=0.02):
    return AbsolutePoseFix(Pose(x, y), error, 0.0, sim_time)


def _estimate(x=1.0, y=1.0, sim_time=0.9, uncertainty=0.03):
    return PoseEstimate(
        Pose(x, y), uncertainty, PoseSource.FUSED, sim_time
    )


def _service(fix=None, **overrides):
    localizer = MagicMock()
    localizer.locate.return_value = fix
    values = {
        "confirmation_samples": 2,
        "confirmation_tolerance_m": 0.1,
        "max_speed_mps": 1.0,
        "max_acceleration_mps2": 1.0,
        "motion_model_error_m": 0.01,
        "prediction_error_per_m": 0.1,
        "prediction_error_per_s": 0.05,
        "max_fix_age_s": 0.5,
        "max_prediction_uncertainty_m": 0.2,
    }
    values.update(overrides)
    return PoseFusionService(
        HolonomicCommandMotionModel(), localizer, MagicMock(), **values
    ), localizer


def _assess(service, frame, previous=None, status=LocalizationStatus.UNINITIALIZED, dt=0.1, command=None):
    return service.assess(
        frame,
        previous,
        status,
        command or DriveCommand(0.0, 0.0),
        dt,
    )


def test_motion_model_integrates_applied_holonomic_command():
    result = HolonomicCommandMotionModel().predict(
        Pose(1.0, 2.0), DriveCommand(0.5, -0.25), 2.0
    )
    assert result == Pose(2.0, 1.5)


def test_initialization_requires_committed_consistent_fixes():
    service, localizer = _service(_fix(2.0, 3.0))

    first = _assess(service, _frame())
    repeated_without_commit = _assess(service, _frame())
    service.commit_confirmation(first.confirmation)
    localizer.locate.return_value = _fix(2.02, 3.01, sim_time=1.1)
    second = _assess(service, _frame(1.1))

    assert first.status is LocalizationStatus.UNINITIALIZED
    assert first.stop_required
    assert repeated_without_commit.confirmation.consecutive_samples == 1
    assert second.confirmation.confirmed
    assert second.status is LocalizationStatus.TRUSTED
    assert second.estimate.pose == Pose(2.02, 3.01)


def test_normal_nearby_fix_corrects_directly_and_remains_trusted():
    service, _ = _service(_fix(1.08, 1.0))

    assessment = _assess(
        service,
        _frame(),
        previous=_estimate(),
        status=LocalizationStatus.TRUSTED,
    )

    assert assessment.status is LocalizationStatus.TRUSTED
    assert assessment.estimate.pose == Pose(1.08, 1.0)
    assert assessment.estimate.source is PoseSource.FUSED
    assert not assessment.stop_required


def test_short_blackout_predicts_and_grows_uncertainty():
    service, _ = _service(None)
    previous = _estimate()

    assessment = _assess(
        service,
        None,
        previous=previous,
        status=LocalizationStatus.TRUSTED,
        command=DriveCommand(0.5, 0.0),
        dt=0.1,
    )

    assert assessment.status is LocalizationStatus.DEGRADED
    assert assessment.estimate.pose == Pose(1.05, 1.0)
    assert assessment.estimate.uncertainty_m > previous.uncertainty_m
    assert not assessment.stop_required


def test_persistent_blackout_stops_when_uncertainty_budget_is_exceeded():
    service, _ = _service(None, max_prediction_uncertainty_m=0.04)

    assessment = _assess(
        service,
        None,
        previous=_estimate(uncertainty=0.039),
        status=LocalizationStatus.DEGRADED,
        command=DriveCommand(1.0, 0.0),
        dt=0.1,
    )

    assert assessment.status is LocalizationStatus.LOST
    assert assessment.estimate is None
    assert assessment.stop_required


def test_single_gap_beyond_max_fix_age_stops():
    service, _ = _service(None, max_fix_age_s=0.2)
    assessment = _assess(
        service,
        None,
        previous=_estimate(),
        status=LocalizationStatus.TRUSTED,
        dt=0.21,
    )
    assert assessment.status is LocalizationStatus.LOST
    assert assessment.stop_required


def test_transient_distant_outlier_stops_same_tick_but_is_not_adopted():
    service, localizer = _service(_fix(8.0, 8.0))
    previous = _estimate()

    outlier = _assess(
        service,
        _frame(),
        previous=previous,
        status=LocalizationStatus.TRUSTED,
    )
    service.commit_confirmation(outlier.confirmation)
    localizer.locate.return_value = _fix(1.04, 1.0, sim_time=1.1)
    normal = _assess(
        service,
        _frame(1.1),
        previous=previous,
        status=LocalizationStatus.TRUSTED,
    )

    assert outlier.stop_required
    assert outlier.status is LocalizationStatus.LOST
    assert outlier.estimate is None
    assert outlier.divergence_m == pytest.approx(previous.pose.distance_to(Pose(8, 8)))
    assert outlier.events == ()
    assert normal.status is LocalizationStatus.TRUSTED
    assert not normal.confirmation.confirmed


def test_repeated_discontinuous_fix_confirms_relocation():
    service, localizer = _service(_fix(8.0, 8.0))
    previous = _estimate()
    first = _assess(
        service, _frame(), previous, LocalizationStatus.TRUSTED
    )
    service.commit_confirmation(first.confirmation)
    localizer.locate.return_value = _fix(8.02, 8.01, sim_time=1.1)

    second = _assess(
        service, _frame(1.1), previous, LocalizationStatus.LOST
    )

    assert second.confirmation.confirmed
    assert second.status is LocalizationStatus.TRUSTED
    assert second.estimate.pose == Pose(8.02, 8.01)
    assert not second.stop_required


def test_stationary_fix_is_not_called_relocation_when_command_requested_motion():
    service, _ = _service(_fix(1.0, 1.0))

    assessment = _assess(
        service,
        _frame(velocity=(0.0, 0.0)),
        previous=_estimate(),
        status=LocalizationStatus.TRUSTED,
        command=DriveCommand(1.0, 0.0),
    )

    assert assessment.status is LocalizationStatus.TRUSTED
    assert not assessment.stop_required
    assert assessment.divergence_m is None


@pytest.mark.parametrize("bad_dt", [-0.1, float("nan")])
def test_fusion_rejects_invalid_delta_time(bad_dt):
    service, _ = _service(None)
    with pytest.raises(ValueError):
        _assess(service, None, previous=_estimate(), dt=bad_dt)
