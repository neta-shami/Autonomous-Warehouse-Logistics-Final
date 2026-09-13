"""Atomic assess/commit/act tests for one robot agent."""

from unittest.mock import MagicMock

import pytest

from entities.drive_command import DriveCommand
from entities.enums import (
    CardinalDirection,
    CommandGateReason,
    FaultCode,
    FaultSeverity,
    FSMStatus,
    LocalizationStatus,
    ManipulatorAbortResult,
    PoseSource,
    TaskExecutionPhase,
    TaskStatus,
    ZoneType,
)
from entities.events import (
    FaultResetEvent,
    PayloadMismatchEvent,
    RelocalizationFailedEvent,
    RelocalizedEvent,
    RobotFaultedEvent,
)
from entities.fleet_snapshot import FleetRobotView, FleetSnapshot
from entities.health import HealthFinding
from entities.localization_assessment import ConfirmationUpdate, LocalizationAssessment
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from entities.robot_state import RobotState
from entities.runtime_control import TickContext
from entities.sensor_frame import SensorFrame
from entities.task import Task
from entities.warehouse_topology import WarehouseTopology
from use_cases.payload_check import PayloadCheck
from use_cases.payload_presence_tracker import PayloadPresenceTracker
from use_cases.robot_agent import RobotAgent
from use_cases.robot_health_monitor import RobotHealthMonitor
from use_cases.robot_states import (
    FaultedState,
    IdleState,
    ManipulatingState,
    NavigatingState,
    RelocalizingState,
)
from use_cases.waypoint_controller import WaypointController


def _estimate(x=1.0, y=1.0, sim_time=1.0):
    return PoseEstimate(Pose(x, y), 0.03, PoseSource.FUSED, sim_time)


def _localization(estimate, status, stop=False):
    return LocalizationAssessment(
        estimate=estimate,
        status=status,
        stop_required=stop,
        divergence_m=None,
        confirmation=ConfirmationUpdate(None, 0, False),
        events=(),
    )


def _frame(robot_id="r1", sim_time=1.0, sequence=0, payload=-1.0):
    return SensorFrame(
        robot_id=robot_id,
        observed_chassis_velocity=(0.2, -0.1),
        wall_ranges=(
            (CardinalDirection.POS_X, 9.0),
            (CardinalDirection.NEG_X, 1.0),
            (CardinalDirection.POS_Y, 9.0),
            (CardinalDirection.NEG_Y, 1.0),
        ),
        payload_range=payload,
        sim_time=sim_time,
        sequence=sequence,
    )


def _agent(state=None, localization=None):
    state = state or RobotState("r1", None, None, None, None, False, (1, 9))
    fusion = MagicMock()
    fusion.assess.return_value = localization or _localization(
        _estimate(), LocalizationStatus.TRUSTED
    )
    return RobotAgent(
        robot_state=state,
        drive_system=MagicMock(),
        manipulator_system=MagicMock(),
        topology=WarehouseTopology(10, 10, 0.0, 10.0, 0.0, 10.0, {}),
        local_avoidance=MagicMock(),
        path_planner=MagicMock(),
        pose_fusion=fusion,
        health_monitor=RobotHealthMonitor([]),
        waypoint_controller=WaypointController(1.0),
        payload_tracker=PayloadPresenceTracker(confirmation_samples=1),
    )


def _snapshot(agent):
    view = FleetRobotView(
        agent.state.robot_id,
        agent.state.pose_estimate,
        agent.state.observed_chassis_velocity,
        agent.FOOTPRINT_RADIUS_M,
        tuple(agent.path[agent.path_index:]),
        agent.state.localization_status,
        agent.state.active_faults,
        agent.fsm_status,
    )
    return FleetSnapshot(1, 1.0, (view,))


def test_assessment_does_not_mutate_until_atomic_commit():
    agent = _agent()

    assessment = agent.assess(_frame(payload=0.02), sim_time=1.0, dt=0.02)

    assert agent.state.pose_estimate is None
    assert agent.state.observed_chassis_velocity is None
    assert agent.state.payload_present is None
    agent.commit(assessment)
    assert agent.state.pose_estimate == _estimate()
    assert agent.state.observed_chassis_velocity == (0.2, -0.1)
    assert agent.state.payload_present is True
    agent.pose_fusion.commit_confirmation.assert_called_once()


def test_wrong_robot_frame_becomes_unknown_stop_assessment():
    state = RobotState(
        "r1",
        _estimate(sim_time=0.9),
        (0.4, 0.0),
        None,
        False,
        False,
        (1, 9),
        LocalizationStatus.TRUSTED,
    )
    agent = _agent(
        state,
        _localization(None, LocalizationStatus.LOST, stop=True),
    )

    assessment = agent.assess(
        _frame(robot_id="r2", sim_time=1.0), sim_time=1.0, dt=0.1
    )

    assert assessment.stop_required
    assert assessment.observation.observed_chassis_velocity is None
    assert assessment.observation.payload_present is None
    agent.pose_fusion.assess.assert_called_once_with(
        None,
        state.pose_estimate,
        LocalizationStatus.TRUSTED,
        DriveCommand(0.0, 0.0),
        0.1,
    )


def test_apply_and_stop_record_exact_hardware_command():
    state = RobotState(
        "r1", _estimate(), (0.0, 0.0), None, False, False, (1, 9),
        LocalizationStatus.TRUSTED,
    )
    agent = _agent(state)
    command = DriveCommand(0.6, 0.8)

    agent.apply_drive(command)

    agent._drive_system.command_velocity.assert_called_once_with("r1", command)
    assert agent.last_applied_drive_command is command
    agent.stop_drive()
    agent._drive_system.stop.assert_called_once_with("r1")
    assert agent.last_applied_drive_command == DriveCommand(0.0, 0.0)


def test_fleet_hold_stops_without_changing_healthy_state_or_task():
    state = RobotState(
        "r1", _estimate(), (0.0, 0.0), None, False, True, (1, 9),
        LocalizationStatus.TRUSTED,
    )
    agent = _agent(state)
    task = object()
    agent.current_task = task
    agent.transition_to(NavigatingState())
    context = TickContext(
        frame=_frame(),
        fleet_snapshot=_snapshot(agent),
        sim_time=1.0,
        dt=0.02,
        gate_reasons=frozenset({CommandGateReason.FLEET_HOLD}),
    )

    agent.act(context)

    assert agent.fsm_status is FSMStatus.NAVIGATING
    assert agent.current_task is task
    assert agent.state.localization_status is LocalizationStatus.TRUSTED
    agent._drive_system.stop.assert_called_once_with("r1")


def test_local_safety_gate_enters_command_free_relocalizing_state():
    state = RobotState(
        "r1", _estimate(), (0.0, 0.0), None, False, True, (1, 9),
        LocalizationStatus.TRUSTED,
    )
    agent = _agent(state)
    agent.transition_to(NavigatingState())
    context = TickContext(
        frame=None,
        fleet_snapshot=_snapshot(agent),
        sim_time=1.0,
        dt=0.02,
        gate_reasons=frozenset({CommandGateReason.LOCAL_SAFETY}),
    )

    agent.act(context)

    assert isinstance(agent.current_state, RelocalizingState)
    agent._drive_system.stop.assert_called_once_with("r1")


def test_manipulation_is_aborted_under_fleet_hold():
    state = RobotState(
        "r1", _estimate(), (0.0, 0.0), None, False, True, (1, 9),
        LocalizationStatus.TRUSTED,
    )
    agent = _agent(state)
    agent.transition_to(ManipulatingState())
    context = TickContext(
        frame=_frame(),
        fleet_snapshot=_snapshot(agent),
        sim_time=1.0,
        dt=0.02,
        gate_reasons=frozenset({CommandGateReason.FLEET_HOLD}),
    )

    agent.act(context)

    agent.manipulator_system.abort.assert_called_once_with("r1")


def test_irreversible_manipulation_abort_quarantines_task():
    state = RobotState(
        "r1", _estimate(), (0.0, 0.0), "p1", True, True, (1, 9),
        LocalizationStatus.TRUSTED,
    )
    agent = _agent(state)
    task = MagicMock()
    agent.current_task = task
    agent.transition_to(ManipulatingState())
    agent.manipulator_system.abort.return_value = (
        ManipulatorAbortResult.RECONCILIATION_REQUIRED
    )
    context = TickContext(
        _frame(), _snapshot(agent), 1.0, 0.02,
        frozenset({CommandGateReason.FLEET_HOLD}),
    )

    agent.act(context)

    task.update_status.assert_called_once_with(TaskStatus.RECOVERY_REQUIRED)
    assert agent.fsm_status is FSMStatus.FAULTED


def test_confirmed_recovery_discards_stale_path_and_emits_event():
    state = RobotState(
        "r1", _estimate(), (0.0, 0.0), None, False, False, (1, 9),
        LocalizationStatus.TRUSTED,
    )
    agent = _agent(state)
    agent.path = [(8, 8)]
    agent._relocalizing_due_to_loss = True
    agent.state_start_sim_time = 0.5
    context = TickContext(_frame(), _snapshot(agent), 1.0, 0.02, frozenset())

    agent.act(context)

    assert agent.path == []
    assert agent.fsm_status is FSMStatus.IDLE
    assert any(isinstance(event, RelocalizedEvent) for event in agent.event_bus)


def test_relocalization_timeout_faults_before_notification_dispatch():
    agent = _agent()
    snapshot = _snapshot(agent)
    gate = frozenset({CommandGateReason.LOCAL_SAFETY})
    agent.act(TickContext(None, snapshot, 0.0, 0.0, gate))
    agent.act(TickContext(None, snapshot, 2.1, 2.1, gate))

    assert agent.fsm_status is FSMStatus.FAULTED
    assert FaultCode.STATE_TIMEOUT in agent.state.active_faults
    assert any(
        isinstance(event, RelocalizationFailedEvent)
        for event in agent.event_bus
    )


def test_faulted_state_publishes_exactly_one_notification():
    agent = _agent()
    agent.state.active_faults = frozenset({FaultCode.STATE_TIMEOUT})
    from use_cases.robot_states import FaultedState

    agent.transition_to(FaultedState(), 0.0)
    context = TickContext(None, _snapshot(agent), 2.1, 0.1, frozenset())

    agent.act(context)
    agent.act(context)

    assert sum(
        isinstance(event, RobotFaultedEvent) for event in agent.event_bus
    ) == 1


@pytest.mark.parametrize(
    "runtime_state", [IdleState, NavigatingState, ManipulatingState]
)
def test_static_geometry_overlap_is_known_but_unsafe_in_every_state(
    runtime_state,
):
    localization = _localization(
        _estimate(2.0, 2.0), LocalizationStatus.TRUSTED
    )
    agent = _agent(localization=localization)
    agent.topology = WarehouseTopology(
        10, 10, 0.0, 10.0, 0.0, 10.0, {(2, 2): ZoneType.SHELF}
    )
    agent.transition_to(runtime_state())

    assessment = agent.assess(_frame(), 1.0, 0.02)

    assert assessment.localization.status is LocalizationStatus.TRUSTED
    assert any(
        finding.fault is FaultCode.POSITION_PHYSICALLY_INVALID
        for finding in assessment.health_findings
    )
    assert assessment.stop_required


def test_existing_latched_fault_keeps_every_later_tick_stopped():
    state = RobotState(
        "r1", _estimate(), (0.3, 0.0), None, False, True, (1, 9),
        LocalizationStatus.TRUSTED,
        frozenset({FaultCode.DRIVE_STALL}),
    )
    agent = _agent(state)

    assessment = agent.assess(_frame(), 1.0, 0.02)

    assert assessment.stop_required


def test_assign_task_refuses_to_overwrite_active_work():
    agent = _agent(
        RobotState(
            "r1",
            _estimate(),
            (0.0, 0.0),
            None,
            False,
            False,
            (1, 9),
            LocalizationStatus.TRUSTED,
        )
    )
    first = Task.park("park-1", (1, 9))
    second = Task.park("park-2", (1, 9))

    agent.assign_task(first)

    with pytest.raises(RuntimeError, match="already has active work"):
        agent.assign_task(second)
    assert agent.current_task is first


def test_non_latched_health_finding_clears_when_it_is_no_longer_reported():
    agent = _agent()
    finding = HealthFinding(
        FaultCode.SENSOR_INVALID,
        FaultSeverity.STOP,
        True,
        False,
        None,
    )
    agent.health_monitor = MagicMock()
    agent.health_monitor.evaluate.side_effect = [(finding,), ()]

    first = agent.assess(_frame(sequence=0), 1.0, 0.02)
    agent.commit(first)
    second = agent.assess(_frame(sim_time=1.02, sequence=1), 1.02, 0.02)
    agent.commit(second)

    assert FaultCode.SENSOR_INVALID not in agent.state.active_faults
    assert not second.stop_required


def test_non_localization_health_fault_enters_faulted_not_relocalizing():
    state = RobotState(
        "r1", _estimate(), (0.0, 0.0), None, False, True, (1, 9),
        LocalizationStatus.TRUSTED,
        frozenset({FaultCode.DRIVE_STALL}),
    )
    agent = _agent(state)
    agent.transition_to(NavigatingState(), 0.0)

    agent.act(
        TickContext(
            _frame(),
            _snapshot(agent),
            1.0,
            0.02,
            frozenset({CommandGateReason.LOCAL_SAFETY}),
        )
    )

    assert isinstance(agent.current_state, FaultedState)
    assert agent.state.localization_status is LocalizationStatus.TRUSTED


def test_fault_notification_is_only_emitted_on_activation_edge():
    state = RobotState(
        "r1", _estimate(), (0.0, 0.0), "p1", False, True, (1, 9),
        LocalizationStatus.TRUSTED,
    )
    agent = _agent(state)
    agent.health_monitor = RobotHealthMonitor([PayloadCheck()])
    agent.current_task = MagicMock(task_id="t1", phase=TaskExecutionPhase.TO_TARGET)

    first = agent.assess(_frame(sim_time=1.0, sequence=0), 1.0, 0.02)
    agent.commit(first)
    second = agent.assess(_frame(sim_time=1.02, sequence=1), 1.02, 0.02)

    assert sum(isinstance(event, PayloadMismatchEvent) for event in first.events) == 1
    assert not any(isinstance(event, PayloadMismatchEvent) for event in second.events)
    assert second.stop_required


def test_request_fault_reset_clears_only_requested_fault_and_audits_it():
    state = RobotState(
        "r1", _estimate(), (0.0, 0.0), "p1", False, True, (1, 9),
        LocalizationStatus.TRUSTED,
        frozenset({FaultCode.PAYLOAD_MISMATCH, FaultCode.STATE_TIMEOUT}),
    )
    agent = _agent(state)
    agent.transition_to(FaultedState(), 1.0)
    from entities.health import HealthCheckContext

    reset_context = HealthCheckContext(
        state,
        _frame(),
        FSMStatus.FAULTED,
        0.0,
        DriveCommand(0.0, 0.0),
        "t1",
        TaskExecutionPhase.TO_TARGET,
        sim_time=2.0,
        inventory_reconciled=True,
    )

    assert agent.request_fault_reset(FaultCode.PAYLOAD_MISMATCH, reset_context)
    assert agent.state.active_faults == frozenset({FaultCode.STATE_TIMEOUT})
    event = next(event for event in agent.event_bus if isinstance(event, FaultResetEvent))
    assert event.cleared_fault is FaultCode.PAYLOAD_MISMATCH
    assert event.sim_time == pytest.approx(2.0)
    assert agent.fsm_status is FSMStatus.FAULTED
