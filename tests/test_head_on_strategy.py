from unittest.mock import MagicMock

from entities.enums import (
    FSMStatus,
    LocalizationStatus,
    PoseSource,
    TaskStatus,
    TaskType,
)
from entities.events import CollisionStallEvent, TaskCompletedEvent
from entities.fleet_snapshot import FleetRobotView, FleetSnapshot
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from entities.robot_state import RobotState
from entities.task import Task
from use_cases.payload_presence_tracker import PayloadPresenceTracker
from use_cases.robot_agent import RobotAgent
from use_cases.robot_health_monitor import RobotHealthMonitor
from use_cases.robot_registry import RobotRegistry
from use_cases.traffic_strategies.head_on_strategy import HeadOnStrategy
from use_cases.waypoint_controller import WaypointController


def _agent_with_task(path_planner, topology):
    """A real RobotAgent, not a mock, so read-only properties actually bite."""
    state = RobotState(
        "r1",
        PoseEstimate(Pose(2.0, 2.0), 0.03, PoseSource.FUSED, 1.0),
        (1.0, 0.0),
        None,
        False,
        False,
        (0, 0),
        LocalizationStatus.TRUSTED,
    )
    agent = RobotAgent(
        state,
        MagicMock(),
        MagicMock(),
        topology,
        MagicMock(),
        path_planner,
        MagicMock(),
        RobotHealthMonitor([]),
        WaypointController(1.0),
        PayloadPresenceTracker(),
    )
    agent.assign_task(
        Task.transfer("t1", TaskType.RETRIEVE, (2, 2), 0, (5, 5), 0, "p1")
    )
    return agent


def _registry_and_snapshot(agent):
    robots = RobotRegistry()
    robots.register("r1", agent)
    snapshot = FleetSnapshot(
        1,
        1.0,
        (
            FleetRobotView(
                "r1",
                agent.state.pose_estimate,
                agent.state.observed_chassis_velocity,
                agent.FOOTPRINT_RADIUS_M,
                tuple(agent.path),
                agent.state.localization_status,
                agent.state.active_faults,
                agent.fsm_status,
            ),
        ),
    )
    return robots, snapshot


def test_head_on_strategy_detours_when_a_path_exists():
    topology = MagicMock()
    path_planner = MagicMock()
    path_planner.find_path.return_value = [(2, 2), (2, 3), (5, 5)]

    agent = _agent_with_task(path_planner, topology)
    robots, snapshot = _registry_and_snapshot(agent)
    HeadOnStrategy(topology, path_planner).resolve(
        CollisionStallEvent(
            sim_time=1.0,
            robot_id="r1",
            coords=(2.0, 2.0),
            blocking_robot_ids=["r2"],
        ),
        robots,
        snapshot,
    )

    assert agent.path == [(2, 2), (2, 3), (5, 5)]
    assert agent.current_task is not None


def test_head_on_strategy_preserves_task_for_retry_when_temporarily_trapped():
    """A missing detour must hold work for a later snapshot, not lose it."""
    topology = MagicMock()
    path_planner = MagicMock()
    path_planner.find_path.return_value = []  # no detour available

    agent = _agent_with_task(path_planner, topology)
    task = agent.current_task
    robots, snapshot = _registry_and_snapshot(agent)

    HeadOnStrategy(topology, path_planner).resolve(
        CollisionStallEvent(
            sim_time=1.0,
            robot_id="r1",
            coords=(2.0, 2.0),
            blocking_robot_ids=["r2"],
        ),
        robots,
        snapshot,
    )

    assert agent.fsm_status == FSMStatus.NAVIGATING
    assert agent.current_task is task
    assert task.status == TaskStatus.IN_PROGRESS
    assert agent.route_replan_required
    assert not [
        event
        for event in agent.event_bus
        if isinstance(event, TaskCompletedEvent) and not event.success
    ]
