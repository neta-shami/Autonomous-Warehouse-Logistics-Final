from unittest.mock import MagicMock

from entities.enums import (
    FSMStatus,
    LocalizationStatus,
    PoseSource,
    TaskType,
    ZoneType,
)
from entities.events import RobotIdleEvent, TaskReadyEvent
from entities.fleet_snapshot import FleetRobotView, FleetSnapshot
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from entities.robot_state import RobotState
from entities.task import Task
from use_cases.fleet_manager import FleetManager
from use_cases.robot_agent import RobotAgent
from use_cases.robot_registry import RobotRegistry


def _state(robot_id="r1", coords=(1.0, 1.0), home=(1, 9)):
    return RobotState(
        robot_id,
        PoseEstimate(Pose(*coords), 0.03, PoseSource.FUSED, 1.0),
        (0.0, 0.0),
        None,
        False,
        False,
        home,
        LocalizationStatus.TRUSTED,
    )


def _baseline_estimator(distance=1.0):
    estimator = MagicMock()
    estimator.estimate.return_value = distance
    return estimator

def test_handle_task_ready_dispatches_ready_tasks():
    # Setup mocks
    reservation_service = MagicMock()
    task = Task.transfer("t1", TaskType.STORE, (1, 1), 0, (2, 2), 0, "p1")
    reservation_service.ready_queue = [task]

    path_planner = MagicMock()
    path_planner.find_path.return_value = [(1, 1), (1, 2)]
    path_planner.find_path_to_access.return_value = (
        (1, 1),
        [(1, 1), (1, 2)],
    )

    topology = MagicMock()

    allocation_strategy = MagicMock()
    allocation_strategy.allocate_tasks.return_value = {"t1": "r1"}

    fleet_manager = FleetManager(
        reservation_service, path_planner, topology, allocation_strategy,
        _baseline_estimator(),
        RobotRegistry(),
    )

    # Setup agent
    agent = MagicMock(spec=RobotAgent)
    agent.fsm_status = FSMStatus.IDLE
    agent.state = _state(coords=(1.0, 1.0), home=(0, 0))
    agent.path = []
    agent.path_index = 0
    agent.current_task = None
    agent.FOOTPRINT_RADIUS_M = 0.35
    agent.WAYPOINT_TOLERANCE = 0.5
    agent.FINAL_WAYPOINT_TOLERANCE = 0.1
    topology.get_access_point.return_value = (1, 1)

    def mock_assign_task(t):
        agent.current_task = t
    agent.assign_task.side_effect = mock_assign_task

    agents = {"r1": agent}
    fleet_manager.robots.replace(agents.items())

    # Run handle_task_ready
    fleet_manager.handle_task_ready(TaskReadyEvent(sim_time=1.0))

    # Verify allocations and routing were called
    allocation_strategy.allocate_tasks.assert_called_once()
    path_planner.find_path_to_access.assert_called()

    # Verify agent was assigned the task
    agent.assign_task.assert_called_with(task)
    assert agent.current_leg_goal == (1, 1)
    agent.set_path.assert_called_with([(1, 1), (1, 2)])

    # Verify task was removed from ready queue
    assert task not in reservation_service.ready_queue


def test_allocator_cannot_overwrite_one_robot_with_two_tasks():
    reservation_service = MagicMock()
    first = Task.transfer(
        "t1", TaskType.STORE, (1, 1), 0, (2, 2), 0, "p1"
    )
    second = Task.transfer(
        "t2", TaskType.STORE, (1, 1), 0, (3, 3), 0, "p2"
    )
    reservation_service.ready_queue = [first, second]
    path_planner = MagicMock()
    path_planner.find_path.return_value = [(1, 1), (1, 2)]
    path_planner.find_path_to_access.return_value = (
        (1, 1),
        [(1, 1), (1, 2)],
    )
    topology = MagicMock()
    topology.get_access_point.return_value = (1, 1)
    allocation_strategy = MagicMock()
    allocation_strategy.allocate_tasks.return_value = {
        "t1": "r1",
        "t2": "r1",
    }
    manager = FleetManager(
        reservation_service, path_planner, topology, allocation_strategy,
        _baseline_estimator(),
        RobotRegistry(),
    )
    agent = MagicMock(spec=RobotAgent)
    agent.fsm_status = FSMStatus.IDLE
    agent.state = _state()
    agent.current_task = None
    agent.path = []
    agent.path_index = 0
    agent.FOOTPRINT_RADIUS_M = 0.35
    agent.WAYPOINT_TOLERANCE = 0.5
    agent.FINAL_WAYPOINT_TOLERANCE = 0.1

    def assign(task):
        agent.current_task = task
        agent.state.is_busy = True

    agent.assign_task.side_effect = assign
    manager.robots.replace({"r1": agent}.items())

    manager.handle_task_ready(TaskReadyEvent(1.0))

    agent.assign_task.assert_called_once_with(first)
    assert reservation_service.ready_queue == [second]

def test_handle_robot_idle_auto_parks():
    reservation_service = MagicMock()
    path_planner = MagicMock()
    path_planner.find_path.return_value = [(1, 1), (5, 5)]

    topology = MagicMock()
    # Mock topology to return NORMAL_FLOOR for the robot's current position
    topology.get_zone_type.return_value = ZoneType.NORMAL_FLOOR

    allocation_strategy = MagicMock()

    fleet_manager = FleetManager(
        reservation_service, path_planner, topology, allocation_strategy,
        _baseline_estimator(),
        RobotRegistry(),
    )

    agent = MagicMock(spec=RobotAgent)
    agent.fsm_status = FSMStatus.IDLE
    agent.state = _state(home=(5, 5))
    agent.current_task = None

    fleet_manager.robots.replace({"r1": agent}.items())
    event = RobotIdleEvent(
        sim_time=1.0, robot_id="r1", current_coords=(1, 1)
    )
    fleet_manager.handle_robot_idle(event)

    # Verify it was assigned a park task
    assert agent.assign_task.call_count == 1
    assigned_task = agent.assign_task.call_args[0][0]
    assert assigned_task.task_type == TaskType.PARK
    assert assigned_task.target_coords == (5, 5)

    # Verify path was set
    agent.set_path.assert_called_with([(1, 1), (5, 5)])


def test_auto_park_planning_avoids_other_committed_routes():
    reservation_service = MagicMock()
    path_planner = MagicMock()
    path_planner.find_path.return_value = [(1, 1), (5, 5)]
    topology = MagicMock()
    topology.get_zone_type.return_value = ZoneType.NORMAL_FLOOR
    manager = FleetManager(
        reservation_service, path_planner, topology, MagicMock(),
        _baseline_estimator(),
        RobotRegistry(),
    )
    agent = MagicMock(spec=RobotAgent)
    agent.fsm_status = FSMStatus.IDLE
    agent.state = _state(home=(5, 5))
    agent.current_task = None
    manager.robots.replace({"r1": agent}.items())
    other_path = ((3, 3), (3, 4))
    manager.latest_snapshot = FleetSnapshot(
        1,
        1.0,
        (
            FleetRobotView(
                "r1",
                agent.state.pose_estimate,
                (0.0, 0.0),
                0.35,
                (),
                LocalizationStatus.TRUSTED,
                frozenset(),
                FSMStatus.IDLE,
            ),
            FleetRobotView(
                "r2",
                _state("r2").pose_estimate,
                (0.0, 0.0),
                0.35,
                other_path,
                LocalizationStatus.TRUSTED,
                frozenset(),
                FSMStatus.NAVIGATING,
            ),
        ),
    )

    manager.handle_robot_idle(RobotIdleEvent(1.0, "r1", (1, 1)))

    assert path_planner.find_path.call_args.args[3] == (other_path,)


def test_handle_robot_idle_ignores_already_parking():
    reservation_service = MagicMock()
    path_planner = MagicMock()

    topology = MagicMock()
    # Mock topology to return PARKING for the robot's current position
    topology.get_zone_type.return_value = ZoneType.PARKING

    allocation_strategy = MagicMock()

    fleet_manager = FleetManager(
        reservation_service, path_planner, topology, allocation_strategy,
        _baseline_estimator(),
        RobotRegistry(),
    )

    agent = MagicMock(spec=RobotAgent)
    agent.fsm_status = FSMStatus.IDLE
    agent.state = _state(home=(1, 1))
    agent.current_task = None

    fleet_manager.robots.replace({"r1": agent}.items())
    event = RobotIdleEvent(
        sim_time=1.0, robot_id="r1", current_coords=(1, 1)
    )
    fleet_manager.handle_robot_idle(event)

    # Verify it was NOT assigned a park task
    agent.assign_task.assert_not_called()

def test_assign_and_route_task_returns_false_if_no_path():
    reservation_service = MagicMock()
    task = Task.transfer("t1", TaskType.STORE, (1, 1), 0, (2, 2), 0, "p1")
    reservation_service.ready_queue = [task]

    path_planner = MagicMock()
    path_planner.find_path.return_value = [] # Path fails
    path_planner.find_path_to_access.return_value = (None, [])

    topology = MagicMock()

    allocation_strategy = MagicMock()
    allocation_strategy.allocate_tasks.return_value = {"t1": "r1"}

    fleet_manager = FleetManager(
        reservation_service, path_planner, topology, allocation_strategy,
        _baseline_estimator(),
        RobotRegistry(),
    )

    agent = MagicMock(spec=RobotAgent)
    agent.fsm_status = FSMStatus.IDLE
    agent.state = _state(coords=(1.0, 1.0), home=(0, 0))
    agent.path = []
    agent.path_index = 0
    agent.current_task = None
    topology.get_access_point.return_value = (1, 1)

    agents = {"r1": agent}
    fleet_manager.robots.replace(agents.items())

    fleet_manager.handle_task_ready(TaskReadyEvent(sim_time=1.0))

    # Verify agent was NOT assigned the task
    agent.assign_task.assert_not_called()

    # Verify task was NOT removed from ready queue
    assert task in reservation_service.ready_queue


def test_uninitialized_robot_is_excluded_from_allocation():
    reservation_service = MagicMock()
    task = Task.park("t1", (2, 1))
    reservation_service.ready_queue = [task]
    allocation_strategy = MagicMock()
    allocation_strategy.allocate_tasks.return_value = {}
    fleet_manager = FleetManager(
        reservation_service, MagicMock(), MagicMock(), allocation_strategy,
        _baseline_estimator(),
        RobotRegistry(),
    )
    agent = MagicMock(spec=RobotAgent)
    agent.fsm_status = FSMStatus.RELOCALIZING
    agent.state = RobotState("r1", None, None, None, None, False, (1, 9))
    fleet_manager.robots.replace({"r1": agent}.items())

    fleet_manager.handle_task_ready(TaskReadyEvent(sim_time=1.0))

    allocation_strategy.allocate_tasks.assert_called_once_with([task], [])
