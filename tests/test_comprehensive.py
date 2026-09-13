"""
Comprehensive test suite for the warehouse simulation.
Tests entity layer, use case layer, and integration scenarios.
"""
from unittest.mock import MagicMock

from entities.drive_command import DriveCommand
from entities.enums import (
    FSMStatus,
    LocalizationStatus,
    ManipulatorAbortResult,
    ManipulatorResult,
    PoseSource,
    TaskStatus,
    TaskType,
    ZoneType,
)
from entities.fleet_snapshot import FleetRobotView, FleetSnapshot
from entities.inventory_world import InventoryWorld
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from entities.robot_state import RobotState
from entities.task import Task
from entities.warehouse_topology import WarehouseTopology
from interfaces.i_drive_system import IDriveSystem
from interfaces.i_manipulator_system import IManipulatorSystem
from use_cases.astar_pathfinding_strategy import AStarPathfindingStrategy
from use_cases.fleet_manager import FleetManager
from use_cases.greedy_task_allocation import GreedyTaskAllocationStrategy
from use_cases.grid_path_planner import GridPathPlanner
from use_cases.inventory_reservation_service import InventoryReservationService
from use_cases.local_collision_avoidance import LocalCollisionAvoidance
from use_cases.payload_presence_tracker import PayloadPresenceTracker
from use_cases.robot_agent import RobotAgent
from use_cases.robot_health_monitor import RobotHealthMonitor
from use_cases.robot_registry import RobotRegistry
from use_cases.robot_states import IdleState
from use_cases.route_baseline_estimator import RouteBaselineEstimator
from use_cases.waypoint_controller import WaypointController


class MockDrive(IDriveSystem):
    def command_velocity(self, robot_id: str, command: DriveCommand):
        pass
    def stop(self, robot_id: str):
        pass

class MockManip(IManipulatorSystem):
    def pick(self, robot_id: str, expected_payload_id: str, tier: int, approach) -> ManipulatorResult:
        return ManipulatorResult.SUCCEEDED
    def drop(self, robot_id: str, expected_payload_id: str, tier: int, approach) -> ManipulatorResult:
        return ManipulatorResult.SUCCEEDED
    def abort(self, robot_id: str) -> ManipulatorAbortResult:
        return ManipulatorAbortResult.CLEAN

def create_topology():
    return WarehouseTopology(10, 10, 0.0, 10.0, 0.0, 10.0, {})


def create_state(robot_id="r1", coords=(1.0, 1.0), busy=False):
    estimate = PoseEstimate(Pose(*coords), 0.03, PoseSource.FUSED, 1.0)
    return RobotState(
        robot_id,
        estimate,
        (0.0, 0.0),
        None,
        False,
        busy,
        (1, 9),
        LocalizationStatus.TRUSTED,
    )


def create_agent(state, topology=None, path_planner=None):
    topology = topology or create_topology()
    path_planner = path_planner or GridPathPlanner(AStarPathfindingStrategy())
    return RobotAgent(
        state,
        MockDrive(),
        MockManip(),
        topology,
        LocalCollisionAvoidance(),
        path_planner,
        MagicMock(),
        RobotHealthMonitor([]),
        WaypointController(1.0),
        PayloadPresenceTracker(),
    )


def create_snapshot(*states):
    return FleetSnapshot(
        1,
        1.0,
        tuple(
            FleetRobotView(
                state.robot_id,
                state.pose_estimate,
                state.observed_chassis_velocity,
                0.35,
                (),
                state.localization_status,
                state.active_faults,
                FSMStatus.IDLE,
            )
            for state in states
        ),
    )

class TestWarehouseTopology:
    def test_within_bounds(self):
        t = create_topology()
        assert t.is_within_bounds(0, 0)
        assert t.is_within_bounds(9, 9)
        assert not t.is_within_bounds(-1, 0)
        assert not t.is_within_bounds(10, 0)

    def test_navigable_with_shelves(self):
        zone_map = {
            (2, 3): ZoneType.SHELF,
            (4, 5): ZoneType.SHELF
        }
        t = WarehouseTopology(10, 10, 0.0, 10.0, 0.0, 10.0, zone_map)
        assert not t.is_navigable(2, 3)
        assert not t.is_navigable(4, 5)
        assert t.is_navigable(1, 3)
        assert t.is_navigable(3, 5)

class TestRobotState:
    def test_apply_believed_pose(self):
        r = RobotState("r1", None, None, None, None, False, (0, 0))
        estimate = PoseEstimate(
            Pose(3.5, 4.2), 0.03, PoseSource.FUSED, 1.0
        )
        r.apply_localization(estimate, LocalizationStatus.TRUSTED)
        r.apply_observation((0.0, 0.5), False)
        assert r.current_coords == (3.5, 4.2)

class TestTask:
    def test_task_lifecycle(self):
        t = Task.transfer("t1", TaskType.RELOCATE, (2, 3), 1, (1, 0), 0, "p1")
        assert t.status == TaskStatus.PENDING
        t.update_status(TaskStatus.IN_PROGRESS)
        assert t.status == TaskStatus.IN_PROGRESS
        t.update_status(TaskStatus.COMPLETED)
        assert t.status == TaskStatus.COMPLETED

class TestGridPathPlanner:
    def test_straight_path(self):
        t = create_topology()
        p = GridPathPlanner(AStarPathfindingStrategy())
        path = p.find_path((0, 0), (3, 0), t, [])
        assert path[0] == (0, 0)
        assert path[-1] == (3, 0)

    def test_no_path(self):
        zone_map = {
            (1, 0): ZoneType.SHELF,
            (0, 1): ZoneType.SHELF
        }
        t = WarehouseTopology(2, 2, 0.0, 2.0, 0.0, 2.0, zone_map)
        p = GridPathPlanner(AStarPathfindingStrategy())
        path = p.find_path((0, 0), (1, 1), t, [])
        assert len(path) == 0

class TestLocalCollisionAvoidance:
    def test_no_collision(self):
        ca = LocalCollisionAvoidance()
        r1 = create_state("r1", (1.0, 1.0))
        r2 = create_state("r2", (5.0, 1.0))
        status, blockers = ca.check_collisions("r1", create_snapshot(r1, r2))
        assert status is None
        assert not blockers

    def test_stuck_after_timeout(self):
        ca = LocalCollisionAvoidance()
        r1 = create_state("r1", (1.0, 1.0))
        r2 = create_state("r2", (1.5, 1.0))
        status, blockers = ca.check_collisions("r1", create_snapshot(r1, r2))
        assert status == FSMStatus.AVOIDING
        assert "r2" in blockers

class TestRobotAgent:
    def test_idle_state(self):
        r = create_state()
        a = create_agent(r)
        a.transition_to(IdleState())
        assert a.fsm_status == FSMStatus.IDLE

    def test_assign_task(self):
        r = create_state()
        a = create_agent(r)
        t = Task.park("t1", (3, 3))
        a.assign_task(t)
        assert a.fsm_status == FSMStatus.NAVIGATING
        assert t.status == TaskStatus.IN_PROGRESS

class TestFleetManager:
    def test_allocates_task_to_idle_robot(self):
        world = InventoryWorld()
        t = create_topology()
        t.zone_map[(2, 2)] = ZoneType.PARKING
        rs = InventoryReservationService(world, t)
        p = GridPathPlanner(AStarPathfindingStrategy())
        fm = FleetManager(
            rs,
            p,
            t,
            GreedyTaskAllocationStrategy(),
            RouteBaselineEstimator(p, t),
            RobotRegistry(),
        )

        task = Task.park("t1", (2, 2))
        rs.process_new_task(task, 1.0)

        r = create_state()
        a = create_agent(r, t, p)
        a.transition_to(IdleState())
        fm.robots.replace({"r1": a}.items())

        from entities.events import TaskReadyEvent
        fm.handle_task_ready(TaskReadyEvent(sim_time=1.0))

        assert a.fsm_status == FSMStatus.NAVIGATING
        assert a.current_task == task
