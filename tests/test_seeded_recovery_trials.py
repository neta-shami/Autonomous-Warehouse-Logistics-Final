"""Deterministic 100-trial, cross-component displaced-robot recovery gate."""

import random
from unittest.mock import MagicMock

from adapters.fault_injector import FaultInjector
from entities.enums import (
    CardinalDirection,
    CommandGateReason,
    LocalizationStatus,
    PoseSource,
    TaskExecutionPhase,
    TaskType,
)
from entities.events import RelocalizedEvent
from entities.fleet_snapshot import FleetRobotView, FleetSnapshot
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from entities.robot_state import RobotState
from entities.runtime_control import TickContext
from entities.sensor_frame import SensorFrame
from entities.task import Task
from entities.warehouse_topology import WarehouseTopology
from use_cases.continuous_wall_range_localizer import ContinuousWallRangeLocalizer
from use_cases.holonomic_command_motion_model import HolonomicCommandMotionModel
from use_cases.metrics_collector import MetricsCollector
from use_cases.payload_presence_tracker import PayloadPresenceTracker
from use_cases.pose_fusion_service import PoseFusionService
from use_cases.robot_agent import RobotAgent
from use_cases.robot_health_monitor import RobotHealthMonitor
from use_cases.robot_states import IdleState, ManipulatingState
from use_cases.waypoint_controller import WaypointController


class PoseFrameSource:
    def __init__(self, pose):
        self.pose = pose
        self.sequence = 0

    def read_frame(self, robot_id):
        self.sequence += 1
        return SensorFrame(
            robot_id,
            (0.0, 0.0),
            (
                (CardinalDirection.POS_X, 10.0 - self.pose.x),
                (CardinalDirection.NEG_X, self.pose.x),
                (CardinalDirection.POS_Y, 10.0 - self.pose.y),
                (CardinalDirection.NEG_Y, self.pose.y),
            ),
            -1.0,
            self.sequence * 0.02,
            self.sequence,
        )


def _snapshot(agent, sim_time):
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
    return FleetSnapshot(1, sim_time, (view,))


def _agent(old_pose, topology, fusion, phase):
    state = RobotState(
        "r1",
        PoseEstimate(old_pose, 0.02, PoseSource.FUSED, 0.0),
        (0.0, 0.0),
        None,
        False,
        False,
        (1, 9),
        LocalizationStatus.TRUSTED,
    )
    agent = RobotAgent(
        state,
        MagicMock(),
        MagicMock(),
        topology,
        MagicMock(),
        MagicMock(),
        fusion,
        RobotHealthMonitor([]),
        WaypointController(1.0),
        PayloadPresenceTracker(confirmation_samples=1),
    )
    if phase is None:
        agent.transition_to(IdleState(), 0.0)
        return agent

    task = Task.transfer(
        "t1", TaskType.RELOCATE, (1, 1), 0, (8, 8), 0, "p1"
    )
    agent.assign_task(task)
    task.phase = phase
    if phase in (TaskExecutionPhase.TO_TARGET, TaskExecutionPhase.DROPPING):
        state.payload_present = True
        agent.current_leg_goal = task.target_coords
    if phase in (TaskExecutionPhase.PICKING, TaskExecutionPhase.DROPPING):
        agent.transition_to(ManipulatingState(), 0.0)
    return agent


def test_seeded_displacement_set_recovers_all_100_trials_safely():
    """Exercise injection, fusion, agent gating, FSM recovery and KPI pairing."""
    rng = random.Random(20260906)
    phases = (
        None,
        TaskExecutionPhase.TO_SOURCE,
        TaskExecutionPhase.TO_TARGET,
        TaskExecutionPhase.PICKING,
        TaskExecutionPhase.DROPPING,
    )
    safe_outcomes = 0

    for trial in range(100):
        old_pose = Pose(1.0, 1.0)
        new_pose = Pose(rng.uniform(3.0, 9.0), rng.uniform(3.0, 9.0))
        source = PoseFrameSource(old_pose)
        metrics = MetricsCollector()
        injector = FaultInjector(source, metrics, random_seed=trial)
        injector.teleport_robot(
            "r1",
            new_pose,
            0.0,
            lambda _robot_id, pose, source=source: setattr(source, "pose", pose),
        )
        topology = WarehouseTopology(10, 10, 0.0, 10.0, 0.0, 10.0, {})
        fusion = PoseFusionService(
            HolonomicCommandMotionModel(),
            ContinuousWallRangeLocalizer(
                pair_residual_tolerance_m=0.05,
                measurement_error_bound_m=0.01,
            ),
            topology,
        )
        agent = _agent(old_pose, topology, fusion, phases[trial % len(phases)])
        first_stopped = False

        for sample in range(3):
            sim_time = (sample + 1) * 0.02
            assessment = agent.assess(
                injector.read_frame("r1"), sim_time, 0.02
            )
            if sample == 0:
                first_stopped = (
                    assessment.stop_required
                    and assessment.localization.status is LocalizationStatus.LOST
                    and assessment.localization.divergence_m is not None
                )
            agent.commit(assessment)
            gate_reasons = (
                frozenset({CommandGateReason.LOCAL_SAFETY})
                if assessment.stop_required
                else frozenset()
            )
            agent.act(
                TickContext(
                    None,
                    _snapshot(agent, sim_time),
                    sim_time,
                    0.02,
                    gate_reasons,
                )
            )
            metrics.record_command_result(
                sim_time,
                unsafe=bool(gate_reasons),
                command=agent.last_applied_drive_command,
            )

        recovery_events = [
            event for event in agent.event_bus if isinstance(event, RelocalizedEvent)
        ]
        for event in recovery_events:
            metrics.handle_relocalized(event)
        recovered = (
            agent.state.localization_status is LocalizationStatus.TRUSTED
            and agent.state.pose_estimate is not None
            and agent.state.pose_estimate.pose.distance_to(new_pose) < 1e-9
        )
        if (
            first_stopped
            and recovered
            and recovery_events
            and metrics.unsafe_command_batches == 0
            and metrics.automatic_recovery_rate == 1.0
            and metrics.false_relocations == 0
        ):
            safe_outcomes += 1

    assert safe_outcomes == 100
