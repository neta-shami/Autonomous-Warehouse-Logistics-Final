"""Domain invariants for immutable localization value objects."""

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from entities.absolute_pose_fix import AbsolutePoseFix
from entities.drive_command import DriveCommand
from entities.enums import (
    CardinalDirection,
    FaultCode,
    FaultSeverity,
    LocalizationStatus,
    PoseSource,
)
from entities.localization_assessment import (
    ConfirmationUpdate,
    LocalizationAssessment,
)
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from entities.robot_observation import RobotObservation
from entities.sensor_frame import SensorFrame


def _frame(**overrides):
    values = {
        "robot_id": "r1",
        "observed_chassis_velocity": (0.1, -0.2),
        "wall_ranges": (
            (CardinalDirection.POS_X, 8.0),
            (CardinalDirection.NEG_X, 2.0),
        ),
        "payload_range": None,
        "sim_time": 1.5,
        "sequence": 4,
    }
    values.update(overrides)
    return SensorFrame(**values)


@pytest.mark.parametrize("bad_value", [nan, inf, -inf])
def test_pose_rejects_non_finite_coordinates(bad_value):
    with pytest.raises(ValueError, match="finite"):
        Pose(bad_value, 1.0)


def test_pose_is_immutable_and_computes_distance():
    pose = Pose(1.0, 2.0)
    assert pose.distance_to(Pose(4.0, 6.0)) == pytest.approx(5.0)
    with pytest.raises(FrozenInstanceError):
        pose.x = 3.0


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("uncertainty_m", -0.01), ("uncertainty_m", inf), ("sim_time", -1.0)],
)
def test_pose_estimate_rejects_invalid_bounds_and_time(field, bad_value):
    values = {
        "pose": Pose(1.0, 2.0),
        "uncertainty_m": 0.1,
        "source": PoseSource.FUSED,
        "sim_time": 2.0,
    }
    values[field] = bad_value
    with pytest.raises(ValueError):
        PoseEstimate(**values)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("error_bound_m", -0.01),
        ("residual_m", nan),
        ("sim_time", -1.0),
    ],
)
def test_absolute_fix_rejects_invalid_measurements(field, bad_value):
    values = {
        "pose": Pose(1.0, 2.0),
        "error_bound_m": 0.05,
        "residual_m": 0.01,
        "sim_time": 2.0,
    }
    values[field] = bad_value
    with pytest.raises(ValueError):
        AbsolutePoseFix(**values)


def test_sensor_frame_rejects_duplicate_wall_directions():
    with pytest.raises(ValueError, match="Duplicate"):
        _frame(
            wall_ranges=(
                (CardinalDirection.POS_X, 1.0),
                (CardinalDirection.POS_X, 2.0),
            )
        )


@pytest.mark.parametrize("bad_range", [nan, inf, -1.01])
def test_sensor_frame_rejects_invalid_wall_ranges(bad_range):
    with pytest.raises(ValueError, match="wall range"):
        _frame(wall_ranges=((CardinalDirection.POS_X, bad_range),))


def test_sensor_frame_requires_immutable_wall_ranges():
    with pytest.raises(TypeError, match="tuple"):
        _frame(wall_ranges=[(CardinalDirection.POS_X, 1.0)])


def test_sensor_frame_preserves_unknown_payload_and_maps_no_hit_to_none():
    frame = _frame(
        wall_ranges=((CardinalDirection.POS_X, -1.0),),
        payload_range=None,
    )
    assert frame.payload_range is None
    assert frame.range_for(CardinalDirection.POS_X) is None
    assert frame.range_for(CardinalDirection.NEG_X) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"observed_chassis_velocity": (nan, 0.0)},
        {"payload_range": -1.01},
        {"sim_time": inf},
        {"sequence": -1},
    ],
)
def test_sensor_frame_rejects_invalid_channels(overrides):
    with pytest.raises((TypeError, ValueError)):
        _frame(**overrides)


def test_robot_observation_allows_an_all_unknown_sample():
    observation = RobotObservation(None, None, None, None, None)
    assert observation.payload_present is None


def test_robot_observation_requires_time_and_sequence_together():
    with pytest.raises(ValueError, match="together"):
        RobotObservation(None, None, None, 1.0, None)


@pytest.mark.parametrize("component", [nan, inf, -inf])
def test_drive_command_rejects_non_finite_components(component):
    with pytest.raises(ValueError, match="finite"):
        DriveCommand(component, 0.0)


def test_confirmation_cannot_be_confirmed_without_a_candidate():
    with pytest.raises(ValueError, match="candidate"):
        ConfirmationUpdate(None, 1, True)


def test_localization_assessment_uses_immutable_events_and_valid_divergence():
    confirmation = ConfirmationUpdate(None, 0, False)
    with pytest.raises(TypeError, match="tuple"):
        LocalizationAssessment(
            None,
            LocalizationStatus.UNINITIALIZED,
            True,
            None,
            confirmation,
            [],
        )
    with pytest.raises(ValueError, match="divergence"):
        LocalizationAssessment(
            None,
            LocalizationStatus.LOST,
            True,
            -0.1,
            confirmation,
            (),
        )


@pytest.mark.parametrize(
    ("value", "field_name"),
    [
        (PoseEstimate(Pose(1, 2), 0.1, PoseSource.FUSED, 1.0), "sim_time"),
        (AbsolutePoseFix(Pose(1, 2), 0.1, 0.01, 1.0), "residual_m"),
        (_frame(), "sequence"),
        (RobotObservation(None, None, None, None, None), "payload_present"),
        (DriveCommand(0.1, 0.2), "vx"),
        (ConfirmationUpdate(None, 0, False), "confirmed"),
        (
            LocalizationAssessment(
                None,
                LocalizationStatus.UNINITIALIZED,
                True,
                None,
                ConfirmationUpdate(None, 0, False),
                (),
            ),
            "stop_required",
        ),
    ],
)
def test_localization_records_are_frozen(value, field_name):
    with pytest.raises(FrozenInstanceError):
        setattr(value, field_name, None)


def test_required_localization_and_fault_enum_members_exist():
    assert {member.name for member in LocalizationStatus} == {
        "UNINITIALIZED",
        "TRUSTED",
        "DEGRADED",
        "LOST",
    }
    assert {member.name for member in FaultCode} == {
        "DRIVE_STALL",
        "PAYLOAD_MISMATCH",
        "SENSOR_INVALID",
        "STATE_TIMEOUT",
        "MANIPULATOR_FAILURE",
        "POSITION_PHYSICALLY_INVALID",
    }
    assert {member.name for member in FaultSeverity} == {"STOP", "QUARANTINE"}
