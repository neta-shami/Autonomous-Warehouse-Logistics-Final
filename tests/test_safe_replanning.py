"""Continuous-start routing and fleet-containment regression tests."""

from unittest.mock import MagicMock

from entities.enums import (
    FaultCode,
    FSMStatus,
    LocalizationStatus,
    PoseSource,
    TaskExecutionPhase,
    TaskStatus,
    TaskType,
    ZoneType,
)
from entities.events import FaultResetEvent, RelocalizedEvent, RobotFaultedEvent
from entities.fleet_snapshot import FleetRobotView, FleetSnapshot
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from entities.robot_state import RobotState
from entities.task import Task
from entities.warehouse_topology import WarehouseTopology
from use_cases.astar_pathfinding_strategy import AStarPathfindingStrategy
from use_cases.fleet_manager import FleetManager
from use_cases.grid_path_planner import GridPathPlanner
from use_cases.robot_agent import RobotAgent
from use_cases.robot_registry import RobotRegistry
from use_cases.robot_states import NavigatingState
from use_cases.route_baseline_estimator import RouteBaselineEstimator


def _topology(zones=None):
    return WarehouseTopology(10, 10, 0.0, 10.0, 0.0, 10.0, zones or {})


def _planner():
    return GridPathPlanner(AStarPathfindingStrategy())


def _estimate(x, y, uncertainty=0.03):
    return PoseEstimate(Pose(x, y), uncertainty, PoseSource.FUSED, 1.0)


def _view(
    robot_id,
    coords,
    *,
    path=(),
    faults=frozenset(),
    status=LocalizationStatus.TRUSTED,
    fsm=FSMStatus.NAVIGATING,
    uncertainty=0.03,
):
    estimate = None if coords is None else _estimate(*coords, uncertainty)
    return FleetRobotView(
        robot_id,
        estimate,
        (0.0, 0.0),
        0.35,
        tuple(path),
        status,
        faults,
        fsm,
    )


def _state(robot_id="r1", coords=(1.0, 1.0), faults=frozenset()):
    return RobotState(
        robot_id,
        _estimate(*coords),
        (0.0, 0.0),
        "p1",
        False,
        True,
        (1, 9),
        LocalizationStatus.TRUSTED,
        faults,
    )


def _manager(planner=None, topology=None):
    reservation = MagicMock()
    reservation.ready_queue = []
    selected_planner = planner or _planner()
    selected_topology = topology or _topology()
    return FleetManager(
        reservation,
        selected_planner,
        selected_topology,
        MagicMock(),
        RouteBaselineEstimator(selected_planner, selected_topology),
        RobotRegistry(),
    )


def _robot_mock():
    robot = MagicMock(spec=RobotAgent)
    robot.route_replan_required = False
    return robot


def test_pathfinder_accepts_mid_cell_start_via_clear_nearby_anchor():
    path = _planner().find_path(
        Pose(1.35, 1.1),
        (4, 1),
        _topology(),
        active_paths=(),
        blocked_cells=frozenset(),
        footprint_radius_m=0.35,
    )

    assert path[0] == (1, 1)
    assert path[-1] == (4, 1)


def test_pathfinder_refuses_anchor_without_clear_footprint_connector():
    topology = MagicMock(spec=WarehouseTopology)
    topology.grid_width = 10
    topology.grid_height = 10
    topology.is_navigable.return_value = True
    topology.is_pose_feasible.return_value = True
    topology.is_segment_feasible.return_value = False

    path = _planner().find_path(
        Pose(1.4, 1.4),
        (5, 5),
        topology,
        active_paths=(),
        blocked_cells=frozenset(),
        footprint_radius_m=0.35,
    )

    assert path == []


def test_dynamic_blocked_cell_is_never_used_even_as_shorter_route():
    path = _planner().find_path(
        Pose(1.0, 1.0),
        (3, 1),
        _topology(),
        active_paths=(),
        blocked_cells=frozenset({(2, 1)}),
        footprint_radius_m=0.35,
    )

    assert path
    assert (2, 1) not in path


def test_near_shelf_continuous_start_uses_only_clear_connector():
    topology = _topology({(2, 2): ZoneType.SHELF})

    path = _planner().find_path(
        Pose(2.0, 1.2),
        (5, 1),
        topology,
        active_paths=(),
        blocked_cells=frozenset(),
        footprint_radius_m=0.35,
    )

    assert path[0] == (2, 1)
    assert (2, 2) not in path


def test_known_faulted_robot_invalidates_intersecting_route_for_one_tick():
    planner = MagicMock()
    planner.find_path.return_value = [(1, 1), (1, 2), (3, 2)]
    planner.find_path_to_access.return_value = (
        (3, 2),
        [(1, 1), (1, 2), (3, 2)],
    )
    manager = _manager(planner)
    agent = _robot_mock()
    agent.state = _state()
    agent.current_leg_goal = (3, 2)
    agent.current_task = Task.transfer(
        "t1", TaskType.RELOCATE, (2, 2), 0, (4, 2), 0, "p1"
    )
    agent.path = [(1, 1), (2, 1), (3, 1), (3, 2)]
    agent.path_index = 0
    manager.robots.replace({"r1": agent}.items())
    snapshot = FleetSnapshot(
        1,
        1.0,
        (
            _view("r1", (1.0, 1.0), path=agent.path),
            _view(
                "r2",
                (2.0, 1.0),
                faults=frozenset({FaultCode.DRIVE_STALL}),
                fsm=FSMStatus.FAULTED,
            ),
        ),
    )

    manager.update_snapshot(snapshot)

    assert manager.command_holds == frozenset({"r1"})
    agent.set_path.assert_called_once_with([(1, 1), (1, 2), (3, 2)])
    assert (2, 1) in (
        planner.find_path_to_access.call_args.kwargs["blocked_cells"]
    )


def test_route_remains_held_when_no_safe_connector_or_detour_exists():
    planner = MagicMock()
    planner.find_path.return_value = []
    planner.find_path_to_access.return_value = (None, [])
    manager = _manager(planner)
    agent = _robot_mock()
    agent.state = _state()
    agent.current_leg_goal = (3, 1)
    agent.current_task = Task.transfer(
        "t1", TaskType.RELOCATE, (2, 2), 0, (4, 2), 0, "p1"
    )
    agent.path = [(1, 1), (2, 1), (3, 1)]
    agent.path_index = 0
    manager.robots.replace({"r1": agent}.items())
    snapshot = FleetSnapshot(
        1,
        1.0,
        (
            _view("r1", (1.4, 1.0), path=agent.path),
            _view(
                "r2",
                (2.0, 1.0),
                faults=frozenset({FaultCode.DRIVE_STALL}),
                fsm=FSMStatus.FAULTED,
            ),
        ),
    )

    manager.update_snapshot(snapshot)
    manager.update_snapshot(snapshot)

    assert manager.command_holds == frozenset({"r1"})
    assert planner.find_path_to_access.call_count == 2
    agent.set_path.assert_not_called()


def test_relocalized_robot_replans_to_preserved_leg_not_task_source():
    planner = MagicMock()
    planner.find_path.return_value = [(4, 4), (5, 4)]
    planner.find_path_to_access.return_value = (
        (5, 4),
        [(4, 4), (5, 4)],
    )
    manager = _manager(planner)
    agent = _robot_mock()
    agent.state = _state(coords=(4.2, 4.0))
    task = Task.transfer(
        "t1", TaskType.STORE, (1, 1), 0, (6, 4), 0, "p1"
    )
    task.phase = TaskExecutionPhase.TO_TARGET
    agent.current_task = task
    agent.current_leg_goal = (5, 4)
    manager.robots.replace({"r1": agent}.items())
    manager.update_snapshot(
        FleetSnapshot(1, 2.0, (_view("r1", (4.2, 4.0), path=()),))
    )

    manager.handle_relocalized(
        RelocalizedEvent(2.0, "r1", Pose(4.2, 4.0), 0.4)
    )

    assert planner.find_path_to_access.call_args.args[1] == (6, 4)
    agent.set_path.assert_called_once_with([(4, 4), (5, 4)])


def test_pre_pick_fault_requeues_task_without_releasing_lease():
    manager = _manager()
    agent = _robot_mock()
    agent.state = _state()
    task = Task.transfer(
        "t1", TaskType.STORE, (1, 1), 0, (3, 3), 0, "p1"
    )
    task.lease_generation = 7
    task.phase = TaskExecutionPhase.TO_SOURCE
    agent.current_task = task
    manager.robots.replace({"r1": agent}.items())

    manager.handle_robot_faulted(
        RobotFaultedEvent(
            2.0,
            "r1",
        )
    )

    assert task.status is TaskStatus.QUEUED
    assert task.lease_generation == 7
    assert task in manager.reservation_service.ready_queue
    agent.release_task_for_requeue.assert_called_once_with(task)


def test_loaded_fault_is_quarantined_and_never_requeued_from_source():
    manager = _manager()
    agent = _robot_mock()
    agent.state = _state()
    agent.state.payload_present = True
    task = Task.transfer(
        "t1", TaskType.STORE, (1, 1), 0, (3, 3), 0, "p1"
    )
    task.phase = TaskExecutionPhase.TO_TARGET
    agent.current_task = task
    manager.robots.replace({"r1": agent}.items())

    manager.handle_robot_faulted(
        RobotFaultedEvent(
            2.0,
            "r1",
        )
    )

    assert task.status is TaskStatus.RECOVERY_REQUIRED
    assert task not in manager.reservation_service.ready_queue
    agent.release_task_for_requeue.assert_not_called()


def test_drive_reset_replans_and_resumes_a_retained_loaded_task():
    planner = MagicMock()
    planner.find_path.return_value = [(4, 4), (5, 4)]
    planner.find_path_to_access.return_value = (
        (5, 4),
        [(4, 4), (5, 4)],
    )
    manager = _manager(planner)
    agent = _robot_mock()
    agent.state = _state(coords=(4.0, 4.0))
    agent.state.payload_present = True
    agent.state.replace_faults(frozenset())
    task = Task.transfer(
        "t1", TaskType.STORE, (1, 1), 0, (6, 4), 0, "p1"
    )
    task.phase = TaskExecutionPhase.TO_TARGET
    task.update_status(TaskStatus.RECOVERY_REQUIRED)
    agent.current_task = task
    agent.current_leg_goal = (5, 4)
    manager.robots.replace({"r1": agent}.items())
    manager.update_snapshot(
        FleetSnapshot(1, 2.0, (_view("r1", (4.0, 4.0), path=()),))
    )

    manager.handle_fault_reset(
        FaultResetEvent(2.0, "r1", FaultCode.DRIVE_STALL)
    )

    assert task.status is TaskStatus.IN_PROGRESS
    agent.set_path.assert_called_once_with([(4, 4), (5, 4)])
    agent.transition_to.assert_called_once()
    assert isinstance(agent.transition_to.call_args.args[0], NavigatingState)


def test_reconciled_failed_drop_resumes_with_loaded_payload():
    planner = MagicMock()
    planner.find_path_to_access.return_value = ((5, 4), [(4, 4), (5, 4)])
    manager = _manager(planner)
    agent = _robot_mock()
    agent.state = _state(coords=(4.0, 4.0))
    agent.state.payload_present = True
    agent.state.replace_faults(frozenset())
    task = Task.transfer(
        "t1", TaskType.RELOCATE, (2, 2), 0, (6, 4), 0, "p1"
    )
    task.phase = TaskExecutionPhase.DROPPING
    task.update_status(TaskStatus.RECOVERY_REQUIRED)
    agent.current_task = task
    agent.current_leg_goal = (5, 4)
    agent.FOOTPRINT_RADIUS_M = 0.35
    manager.robots.replace({"r1": agent}.items())
    manager.update_snapshot(
        FleetSnapshot(1, 2.0, (_view("r1", (4.0, 4.0), path=()),))
    )

    manager.handle_fault_reset(
        FaultResetEvent(2.0, "r1", FaultCode.MANIPULATOR_FAILURE)
    )

    assert task.phase is TaskExecutionPhase.TO_TARGET
    assert task.status is TaskStatus.IN_PROGRESS
    assert agent.current_leg_goal == (5, 4)
    agent.set_path.assert_called_once_with([(4, 4), (5, 4)])
    assert isinstance(agent.transition_to.call_args.args[0], NavigatingState)


def test_inconsistent_reconciliation_restores_fault_for_another_attempt():
    manager = _manager(MagicMock())
    agent = _robot_mock()
    agent.state = _state(coords=(4.0, 4.0))
    agent.state.payload_present = False
    agent.state.replace_faults(frozenset())
    task = Task.transfer(
        "t1", TaskType.RELOCATE, (2, 2), 0, (6, 4), 0, "p1"
    )
    task.phase = TaskExecutionPhase.DROPPING
    agent.current_task = task
    agent.current_leg_goal = (5, 4)
    agent.FOOTPRINT_RADIUS_M = 0.35
    manager.robots.replace({"r1": agent}.items())
    manager.update_snapshot(
        FleetSnapshot(1, 2.0, (_view("r1", (4.0, 4.0), path=()),))
    )

    manager.handle_fault_reset(
        FaultResetEvent(2.0, "r1", FaultCode.MANIPULATOR_FAILURE)
    )

    assert FaultCode.MANIPULATOR_FAILURE in agent.state.active_faults
    assert task.status is TaskStatus.RECOVERY_REQUIRED
    assert agent.transition_to.call_args.args[0].fsm_status is FSMStatus.FAULTED


def test_faulted_robot_is_excluded_from_task_allocation():
    manager = _manager()
    task = Task.park("park", (4, 4))
    manager.reservation_service.ready_queue = [task]
    agent = _robot_mock()
    agent.fsm_status = FSMStatus.FAULTED
    agent.state = _state(faults=frozenset({FaultCode.DRIVE_STALL}))
    manager.robots.replace({"r1": agent}.items())

    manager.handle_task_ready(MagicMock())

    manager.allocation_strategy.allocate_tasks.assert_called_once_with(
        [task], []
    )
    agent.assign_task.assert_not_called()
