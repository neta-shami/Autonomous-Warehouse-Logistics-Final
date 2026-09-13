"""Independent health-rule and latched-fault regression tests."""

from dataclasses import replace

import pytest

from entities.drive_command import DriveCommand
from entities.enums import (
    FaultCode,
    FSMStatus,
    LocalizationStatus,
    PoseSource,
    TaskExecutionPhase,
)
from entities.events import PayloadMismatchEvent
from entities.health import HealthCheckContext
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from entities.robot_state import RobotState
from use_cases.payload_check import PayloadCheck
from use_cases.robot_health_monitor import RobotHealthMonitor
from use_cases.stall_check import StallCheck
from use_cases.state_timeout_check import StateTimeoutCheck


def _state(*, velocity=(0.0, 0.0), payload=False):
    return RobotState(
        "r1",
        PoseEstimate(Pose(1.0, 1.0), 0.02, PoseSource.ABSOLUTE_FIX, 1.0),
        velocity,
        "p1",
        payload,
        True,
        (1, 9),
        LocalizationStatus.TRUSTED,
    )


def _context(**overrides):
    values = {
        "robot": _state(),
        "frame": None,
        "fsm_status": FSMStatus.NAVIGATING,
        "state_duration_s": 0.0,
        "last_applied_drive_command": DriveCommand(0.5, 0.0),
        "current_task_id": "t1",
        "task_phase": TaskExecutionPhase.TO_SOURCE,
        "sim_time": 1.0,
        "payload_present": False,
    }
    values.update(overrides)
    return HealthCheckContext(**values)


def test_stall_requires_persistent_commanded_motion_without_observed_motion():
    check = StallCheck(persistence_s=0.5)

    assert check.evaluate(_context(sim_time=1.0)) is None
    assert check.evaluate(_context(sim_time=1.49)) is None
    finding = check.evaluate(_context(sim_time=1.5))

    assert finding.fault is FaultCode.DRIVE_STALL
    assert finding.stop_required and finding.latched
    assert finding.event is None


@pytest.mark.parametrize(
    "fsm_status",
    [FSMStatus.IDLE, FSMStatus.AVOIDING, FSMStatus.RELOCALIZING],
)
def test_stall_ignores_intentional_stationary_states(fsm_status):
    check = StallCheck(persistence_s=0.0)

    assert check.evaluate(_context(fsm_status=fsm_status)) is None


def test_stall_ignores_a_recorded_stop_and_does_not_change_localization():
    robot = _state()
    check = StallCheck(persistence_s=0.0)

    finding = check.evaluate(
        _context(
            robot=robot,
            last_applied_drive_command=DriveCommand(0.0, 0.0),
        )
    )

    assert finding is None
    assert robot.localization_status is LocalizationStatus.TRUSTED
    assert robot.pose_estimate is not None


@pytest.mark.parametrize(
    ("phase", "present"),
    [
        (TaskExecutionPhase.TO_SOURCE, True),
        (TaskExecutionPhase.TO_TARGET, False),
    ],
)
def test_payload_check_is_phase_aware(phase, present):
    finding = PayloadCheck().evaluate(
        _context(task_phase=phase, payload_present=present)
    )

    assert finding.fault is FaultCode.PAYLOAD_MISMATCH
    assert finding.stop_required and finding.latched
    assert isinstance(finding.event, PayloadMismatchEvent)


@pytest.mark.parametrize(
    "phase", [TaskExecutionPhase.PICKING, TaskExecutionPhase.DROPPING]
)
def test_payload_check_ignores_contact_transitions(phase):
    assert (
        PayloadCheck().evaluate(
            _context(task_phase=phase, payload_present=True)
        )
        is None
    )


def test_payload_check_treats_unknown_presence_as_inconclusive():
    assert PayloadCheck().evaluate(_context(payload_present=None)) is None


def test_monitor_preserves_simultaneous_independent_findings():
    stall = StallCheck(persistence_s=0.0)
    monitor = RobotHealthMonitor([stall, PayloadCheck()])

    findings = monitor.evaluate(
        _context(task_phase=TaskExecutionPhase.TO_TARGET, payload_present=False)
    )

    assert {finding.fault for finding in findings} == {
        FaultCode.DRIVE_STALL,
        FaultCode.PAYLOAD_MISMATCH,
    }


def test_state_timeout_uses_boundary_and_only_configured_states():
    check = StateTimeoutCheck({FSMStatus.NAVIGATING: 3.0})

    assert check.evaluate(_context(state_duration_s=2.999)) is None
    finding = check.evaluate(_context(state_duration_s=3.0))
    assert finding.fault is FaultCode.STATE_TIMEOUT
    assert finding.stop_required and finding.latched
    assert check.evaluate(
        _context(fsm_status=FSMStatus.IDLE, state_duration_s=1000.0)
    ) is None


def test_reset_policy_requires_fault_specific_evidence():
    monitor = RobotHealthMonitor([])
    context = _context()

    assert not monitor.can_clear(FaultCode.DRIVE_STALL, context)
    assert monitor.can_clear(
        FaultCode.DRIVE_STALL,
        replace(
            context,
            drive_diagnostic_passed=True,
            drive_cause_removed=True,
        ),
    )
    assert not monitor.can_clear(FaultCode.PAYLOAD_MISMATCH, context)
    assert monitor.can_clear(
        FaultCode.PAYLOAD_MISMATCH,
        replace(context, inventory_reconciled=True),
    )
    assert not monitor.can_clear(FaultCode.POSITION_PHYSICALLY_INVALID, context)
    assert monitor.can_clear(
        FaultCode.POSITION_PHYSICALLY_INVALID,
        replace(context, position_feasible=True),
    )


@pytest.mark.parametrize(
    ("constructor", "arguments"),
    [
        (StallCheck, {"persistence_s": -0.1}),
        (StallCheck, {"velocity_threshold": -0.1}),
        (StateTimeoutCheck, {"timeouts": {FSMStatus.NAVIGATING: 0.0}}),
    ],
)
def test_health_check_configuration_rejects_invalid_thresholds(
    constructor, arguments
):
    with pytest.raises((TypeError, ValueError)):
        constructor(**arguments)
