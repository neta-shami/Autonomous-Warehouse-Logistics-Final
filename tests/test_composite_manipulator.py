from unittest.mock import patch

import pytest

from entities.enums import (
    CardinalDirection,
    ManipulatorAbortResult,
    ManipulatorResult,
)
from use_cases.composite_manipulator import CompositeManipulator


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def get_time(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class RecordingHardware:
    def __init__(self):
        self.calls = []
        self.ready = {"turret": True, "lift": True, "extend": True}

    def aim_turret(self, robot_id, approach):
        self.calls.append(("aim", robot_id, approach))

    def set_lift_height(self, robot_id, height):
        self.calls.append(("lift", robot_id, height))

    def extend_arm(self, robot_id, extension):
        self.calls.append(("extend", robot_id, extension))

    def grip(self, robot_id, expected_payload_id, engage):
        self.calls.append(("grip", robot_id, expected_payload_id, engage))

    def joint_at_target(
        self, robot_id, joint_name, target, tolerance
    ):
        self.calls.append(
            ("feedback", robot_id, joint_name, target, tolerance)
        )
        return self.ready[joint_name]


def make_manipulator():
    hardware = RecordingHardware()
    clock = FakeClock()
    manipulator = CompositeManipulator(
        hardware, hardware, hardware, hardware, hardware, clock
    )
    return manipulator, hardware, clock


def test_pick_advances_only_after_feedback_and_returns_typed_success():
    manipulator, hardware, _ = make_manipulator()

    results = [
        manipulator.pick("r1", "p1", 0, CardinalDirection.NEG_Y)
        for _ in range(5)
    ]

    assert results[:-1] == [ManipulatorResult.IN_PROGRESS] * 4
    assert results[-1] is ManipulatorResult.SUCCEEDED
    assert ("grip", "r1", "p1", True) in hardware.calls
    assert "r1" not in manipulator.states


def test_drop_releases_payload_and_uses_upper_tier_height():
    manipulator, hardware, _ = make_manipulator()

    results = [
        manipulator.drop("r1", "p1", 1, CardinalDirection.POS_X)
        for _ in range(5)
    ]

    assert results[-1] is ManipulatorResult.SUCCEEDED
    assert ("lift", "r1", 0.78) in hardware.calls
    assert ("grip", "r1", "p1", False) in hardware.calls


def test_lift_feedback_blocks_extension_and_grip():
    manipulator, hardware, clock = make_manipulator()
    hardware.ready["lift"] = False

    for _ in range(4):
        result = manipulator.pick("r1", "p1", 0, CardinalDirection.NEG_X)
        clock.advance(0.1)

    assert result is ManipulatorResult.IN_PROGRESS
    assert not any(call[0] == "extend" for call in hardware.calls)
    assert not any(call[0] == "grip" for call in hardware.calls)


def test_operation_times_out_using_simulation_time():
    manipulator, hardware, clock = make_manipulator()
    hardware.ready["lift"] = False
    with patch(
        "time.time",
        side_effect=AssertionError("timeout read wall-clock time"),
    ):
        manipulator.pick("r1", "p1", 0, CardinalDirection.POS_X)
        clock.advance(manipulator.OPERATION_TIMEOUT_SECONDS + 0.01)
        result = manipulator.pick("r1", "p1", 0, CardinalDirection.POS_X)

    assert result is ManipulatorResult.FAILED
    assert "r1" in manipulator.states
    assert manipulator.abort("r1") is ManipulatorAbortResult.CLEAN
    assert "r1" not in manipulator.states


def test_abort_before_grip_is_clean_and_retracts():
    manipulator, hardware, _ = make_manipulator()
    hardware.ready["lift"] = False
    manipulator.pick("r1", "p1", 0, CardinalDirection.POS_X)

    result = manipulator.abort("r1")

    assert result is ManipulatorAbortResult.CLEAN
    assert ("extend", "r1", 0.0) in hardware.calls
    assert ("lift", "r1", 0.0) in hardware.calls
    assert "r1" not in manipulator.states


@pytest.mark.parametrize("method_name", ["pick", "drop"])
def test_abort_after_grip_or_release_requires_reconciliation(method_name):
    manipulator, hardware, _ = make_manipulator()
    method = getattr(manipulator, method_name)
    for _ in range(3):
        method("r1", "p1", 0, CardinalDirection.POS_Y)

    result = manipulator.abort("r1")

    assert result is ManipulatorAbortResult.RECONCILIATION_REQUIRED
    assert any(call[0] == "grip" for call in hardware.calls)
    assert "r1" not in manipulator.states


def test_abort_without_operation_is_clean():
    manipulator, _, _ = make_manipulator()
    assert manipulator.abort("r1") is ManipulatorAbortResult.CLEAN


def test_invalid_tier_fails_before_commanding_hardware():
    manipulator, hardware, _ = make_manipulator()

    with pytest.raises(ValueError, match="Unsupported shelf tier"):
        manipulator.pick("r1", "p1", 2, CardinalDirection.POS_X)

    assert hardware.calls == []


def test_operation_parameters_cannot_change_mid_sequence():
    manipulator, _, _ = make_manipulator()
    manipulator.pick("r1", "p1", 0, CardinalDirection.POS_X)

    with pytest.raises(ValueError, match="already has an active"):
        manipulator.pick("r1", "p1", 1, CardinalDirection.POS_X)


def test_each_robot_keeps_an_independent_sequence():
    manipulator, hardware, _ = make_manipulator()
    hardware.ready["lift"] = False

    manipulator.pick("r1", "p1", 0, CardinalDirection.POS_X)
    manipulator.pick("r2", "p2", 1, CardinalDirection.NEG_X)

    assert set(manipulator.states) == {"r1", "r2"}
