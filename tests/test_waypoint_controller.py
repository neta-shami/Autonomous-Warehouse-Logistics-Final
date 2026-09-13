"""Pure tests for belief-only waypoint velocity decisions."""

import math

import pytest

from entities.drive_command import DriveCommand
from entities.pose import Pose
from use_cases.holonomic_command_motion_model import HolonomicCommandMotionModel
from use_cases.waypoint_controller import WaypointController


def test_saturates_vector_speed_once_before_hardware_boundary():
    controller = WaypointController(max_speed_mps=1.0, proportional_gain=2.0)

    command = controller.command_for(
        believed_pose=Pose(0.0, 0.0),
        waypoint=(3.0, 4.0),
        arrival_tolerance_m=0.1,
    )

    assert command.vx == pytest.approx(0.6)
    assert command.vy == pytest.approx(0.8)
    predicted = HolonomicCommandMotionModel().predict(
        Pose(0.0, 0.0), command, 0.5
    )
    assert predicted.x == pytest.approx(0.3)
    assert predicted.y == pytest.approx(0.4)


def test_near_waypoint_uses_proportional_speed_without_overshoot():
    controller = WaypointController(max_speed_mps=2.0, proportional_gain=2.0)

    command = controller.command_for(Pose(1.0, 1.0), (1.3, 1.4), 0.1)

    assert command.vx == pytest.approx(0.6)
    assert command.vy == pytest.approx(0.8)


def test_arrival_returns_an_explicit_zero_command():
    controller = WaypointController(max_speed_mps=1.0)

    command = controller.command_for(Pose(1.0, 1.0), (1.05, 1.0), 0.1)

    assert command == DriveCommand(0.0, 0.0)


def test_rejects_invalid_configuration_and_inputs():
    with pytest.raises(ValueError, match="max_speed_mps"):
        WaypointController(max_speed_mps=0.0)
    with pytest.raises(ValueError, match="proportional_gain"):
        WaypointController(max_speed_mps=1.0, proportional_gain=-1.0)

    controller = WaypointController(max_speed_mps=1.0)
    with pytest.raises(TypeError, match="believed_pose"):
        controller.command_for((0.0, 0.0), (1.0, 1.0), 0.1)
    with pytest.raises(ValueError, match="waypoint"):
        controller.command_for(Pose(0.0, 0.0), (math.nan, 1.0), 0.1)
    with pytest.raises(ValueError, match="arrival_tolerance_m"):
        controller.command_for(Pose(0.0, 0.0), (1.0, 1.0), -0.1)
