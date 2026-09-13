"""End-to-end proof that successful task events match physical package motion."""

from math import dist
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from adapters.cli_parser import CLIParser
from adapters.mujoco_controller import MuJoCoController
from adapters.mujoco_environment_adapter import MuJoCoEnvironmentAdapter
from adapters.mujoco_ground_truth_probe import MuJoCoGroundTruthProbe
from adapters.sensor_adapter import MuJoCoSensorAdapter
from adapters.simulation_loader import SimulationLoader
from adapters.simulation_orchestrator import SimulationOrchestrator
from entities.enums import (
    FaultCode,
    ManipulatorAbortResult,
    ManipulatorResult,
    SnapshotType,
    TaskExecutionPhase,
    TaskStatus,
    TrafficPattern,
)
from entities.events import PackageDroppedEvent
from entities.inventory_world import InventoryWorld
from use_cases.astar_pathfinding_strategy import AStarPathfindingStrategy
from use_cases.composite_manipulator import CompositeManipulator
from use_cases.fleet_manager import FleetManager
from use_cases.greedy_task_allocation import GreedyTaskAllocationStrategy
from use_cases.grid_path_planner import GridPathPlanner
from use_cases.inventory_reservation_service import InventoryReservationService
from use_cases.metrics_collector import MetricsCollector
from use_cases.robot_registry import RobotRegistry
from use_cases.route_baseline_estimator import RouteBaselineEstimator
from use_cases.traffic_manager import TrafficManager
from use_cases.traffic_strategies.head_on_strategy import HeadOnStrategy
from use_cases.traffic_strategies.rear_end_strategy import RearEndStrategy
from use_cases.traffic_strategies.snapshot_detour_strategy import SnapshotDetourStrategy
from use_cases.warehouse_scenario_manager import WarehouseScenarioManager


def _package_position(model, data, package_id):
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"package_{package_id}_free"
    )
    assert joint_id >= 0, f"Missing physical package joint for {package_id}"
    qpos_address = model.jnt_qposadr[joint_id]
    return tuple(float(value) for value in data.qpos[qpos_address:qpos_address + 3])


def test_demo_deliveries_correspond_to_physical_package_motion():
    inventory = InventoryWorld()
    path_planner = GridPathPlanner(AStarPathfindingStrategy())
    xml_path = Path(__file__).parents[1] / "warehouse.xml"
    model, data, topology = SimulationLoader(str(xml_path)).load()
    reservations = InventoryReservationService(inventory, topology)
    metrics = MetricsCollector(MuJoCoGroundTruthProbe(model, data))
    environment = MuJoCoEnvironmentAdapter(model, data)
    controller = MuJoCoController(model, data)
    manipulator = CompositeManipulator(
        controller,
        controller,
        controller,
        controller,
        controller,
        environment,
    )
    scenario = WarehouseScenarioManager(environment, reservations)
    robot_registry = RobotRegistry()
    fleet = FleetManager(
        reservations,
        path_planner,
        topology,
        GreedyTaskAllocationStrategy(),
        RouteBaselineEstimator(path_planner, topology),
        robot_registry,
    )
    traffic = TrafficManager(topology, path_planner, robot_registry)
    traffic.set_strategy(
        TrafficPattern.REAR_END, RearEndStrategy(topology, path_planner)
    )
    traffic.set_strategy(
        TrafficPattern.HEAD_ON, HeadOnStrategy(topology, path_planner)
    )
    detour_strategy = SnapshotDetourStrategy(topology, path_planner)
    traffic.set_strategy(TrafficPattern.CROSSING, detour_strategy)
    traffic.set_strategy(TrafficPattern.STUCK_BETWEEN, detour_strategy)
    orchestrator = SimulationOrchestrator(
        environment=environment,
        scenario_manager=scenario,
        sensor_adapter=MuJoCoSensorAdapter(model, data),
        controller=controller,
        attachment_synchronizer=controller,
        manipulator=manipulator,
        topology=topology,
        fleet_manager=fleet,
        traffic_manager=traffic,
        metrics=metrics,
        path_planner=path_planner,
        robot_registry=robot_registry,
    )
    orchestrator.register_robot("r1", (1.0, 9.0))
    orchestrator.register_robot("r2", (9.0, 9.0))

    physical = inventory.get_snapshot(SnapshotType.PHYSICAL)
    reserved = inventory.get_snapshot(SnapshotType.RESERVED)
    physical.set_cell_occupancy(2, 5, 0, "p2")
    reserved.set_cell_occupancy(2, 5, 0, "p2")

    drop_positions = {}
    package_by_task = {"p1": "p1", "rt_p2": "p2"}

    def record_drop(event):
        drop_positions[event.task.task_id] = _package_position(
            model, data, package_by_task[event.task.task_id]
        )

    orchestrator.event_dispatcher.subscribe(PackageDroppedEvent, record_drop)

    assert orchestrator.spawn_package("p1", (2, 2), 1)
    for task in CLIParser.parse_commands(["RETRIEVE p2 2,5 0"], topology):
        reservations.process_new_task(task, environment.get_time())

    for _ in range(5_000):
        orchestrator.step()
        if drop_positions.keys() >= {"p1", "rt_p2"}:
            break

    assert drop_positions.keys() >= {"p1", "rt_p2"}
    stored = drop_positions["p1"]
    retrieved = drop_positions["rt_p2"]
    assert dist(stored[:2], (2.0, 2.0)) <= 0.35
    assert stored[2] == pytest.approx(0.86, abs=0.08)
    assert dist(retrieved[:2], (9.0, 8.0)) <= 0.35
    assert retrieved[2] == pytest.approx(0.46, abs=0.08)
    assert dist(retrieved[:2], (2.0, 5.0)) > 1.0
    assert metrics.total_throughput == 2
    assert metrics.total_failed_tasks == 0
    assert metrics.total_actual_collisions == 0
    assert metrics.total_distance_driven <= 49.52 * 1.10
    assert metrics.mean_localization_error_m <= 0.10
    assert metrics.peak_localization_error_m <= 0.30
    assert metrics.grid_cell_accuracy >= 0.99
    assert metrics.absolute_fix_availability >= 0.99
    assert metrics.unsafe_command_batches == 0
    assert metrics.orphan_lease_count == 0
    assert metrics.initialization_durations_s
    assert max(metrics.initialization_durations_s.values()) <= 0.25
    assert controller.attached_packages == {}


def test_physical_payload_and_lease_survive_a_clean_pre_release_drop_failure(
    monkeypatch,
):
    """Cover the loaded failure boundary with a real MuJoCo pick."""
    inventory = InventoryWorld()
    xml_path = Path(__file__).parents[1] / "warehouse.xml"
    model, data, topology = SimulationLoader(str(xml_path)).load()
    reservations = InventoryReservationService(inventory, topology)
    planner = GridPathPlanner(AStarPathfindingStrategy())
    environment = MuJoCoEnvironmentAdapter(model, data)
    controller = MuJoCoController(model, data)
    manipulator = CompositeManipulator(
        controller,
        controller,
        controller,
        controller,
        controller,
        environment,
    )
    metrics = MetricsCollector(MuJoCoGroundTruthProbe(model, data))
    robot_registry = RobotRegistry()
    fleet = FleetManager(
        reservations,
        planner,
        topology,
        GreedyTaskAllocationStrategy(),
        RouteBaselineEstimator(planner, topology),
        robot_registry,
    )
    orchestrator = SimulationOrchestrator(
        environment,
        WarehouseScenarioManager(environment, reservations),
        MuJoCoSensorAdapter(model, data),
        controller,
        controller,
        manipulator,
        topology,
        fleet,
        TrafficManager(topology, planner, robot_registry),
        metrics,
        planner,
        robot_registry,
    )
    orchestrator.register_robot("r1", (1.0, 9.0))
    physical = inventory.get_snapshot(SnapshotType.PHYSICAL)
    reserved = inventory.get_snapshot(SnapshotType.RESERVED)
    physical.set_cell_occupancy(2, 5, 0, "p2")
    reserved.set_cell_occupancy(2, 5, 0, "p2")
    task = CLIParser.parse_commands(["RETRIEVE p2 2,5 0"], topology)[0]
    reservations.process_new_task(task, environment.get_time())

    for _ in range(3_000):
        orchestrator.step()
        if task.phase is TaskExecutionPhase.DROPPING:
            break

    assert task.phase is TaskExecutionPhase.DROPPING
    assert controller.attached_packages == {"r1": "package_p2_free"}
    monkeypatch.setattr(
        manipulator, "drop", lambda *_args: ManipulatorResult.FAILED
    )
    monkeypatch.setattr(
        manipulator, "abort", lambda *_args: ManipulatorAbortResult.CLEAN
    )

    orchestrator.step()

    agent = orchestrator.agents["r1"]
    lease = inventory.task_leases[task.task_id]
    assert task.status is TaskStatus.RECOVERY_REQUIRED
    assert agent.current_task is task
    assert FaultCode.MANIPULATOR_FAILURE in agent.state.active_faults
    assert lease.source_released
    assert not lease.target_released
    assert controller.attached_packages == {"r1": "package_p2_free"}
    assert metrics.total_failed_tasks == 0
    assert metrics.orphan_lease_count == 0
