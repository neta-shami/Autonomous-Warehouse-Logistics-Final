"""Single production composition root for the warehouse application."""

from dataclasses import dataclass
from typing import Tuple

import mujoco

from adapters.mujoco_controller import MuJoCoController
from adapters.mujoco_environment_adapter import MuJoCoEnvironmentAdapter
from adapters.mujoco_ground_truth_probe import MuJoCoGroundTruthProbe
from adapters.sensor_adapter import MuJoCoSensorAdapter
from adapters.simulation_loader import SimulationLoader
from adapters.simulation_orchestrator import SimulationOrchestrator
from entities.enums import SnapshotType, TrafficPattern
from entities.inventory_world import InventoryWorld
from entities.warehouse_topology import WarehouseTopology
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
from use_cases.traffic_strategies.snapshot_detour_strategy import (
    SnapshotDetourStrategy,
)
from use_cases.warehouse_scenario_manager import WarehouseScenarioManager


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    home: Tuple[float, float]


@dataclass(frozen=True)
class ApplicationRuntime:
    """Fully wired application plus boundaries needed by entry-point scenarios."""

    model: mujoco.MjModel
    data: mujoco.MjData
    topology: WarehouseTopology
    inventory: InventoryWorld
    reservations: InventoryReservationService
    metrics: MetricsCollector
    environment: MuJoCoEnvironmentAdapter
    controller: MuJoCoController
    scenario: WarehouseScenarioManager
    fleet: FleetManager
    traffic: TrafficManager
    robots: RobotRegistry
    orchestrator: SimulationOrchestrator


DEFAULT_ROBOTS = (
    RobotSpec("r1", (1.0, 9.0)),
    RobotSpec("r2", (9.0, 9.0)),
    RobotSpec("r3", (3.0, 9.0)),
    RobotSpec("r4", (7.0, 9.0)),
)

INITIAL_PACKAGES = (
    (2, 5, 0, "p2"),
    (4, 4, 1, "p3"),
    (6, 6, 0, "p4"),
    (8, 3, 0, "p6"),
)


def build_application(
    xml_path: str = "warehouse.xml",
    robot_specs: Tuple[RobotSpec, ...] = DEFAULT_ROBOTS,
) -> ApplicationRuntime:
    """Construct the application consistently for every production entry point."""
    model, data, topology = SimulationLoader(xml_path).load()
    inventory = InventoryWorld()
    reservations = InventoryReservationService(inventory, topology)
    path_planner = GridPathPlanner(AStarPathfindingStrategy())
    metrics = MetricsCollector(MuJoCoGroundTruthProbe(model, data))
    environment = MuJoCoEnvironmentAdapter(model, data)
    controller = MuJoCoController(model, data)
    scenario = WarehouseScenarioManager(environment, reservations)
    robots = RobotRegistry()
    fleet = FleetManager(
        reservations,
        path_planner,
        topology,
        GreedyTaskAllocationStrategy(),
        RouteBaselineEstimator(path_planner, topology),
        robots,
    )
    traffic = _build_traffic_manager(topology, path_planner, robots)
    orchestrator = SimulationOrchestrator(
        environment=environment,
        scenario_manager=scenario,
        sensor_adapter=MuJoCoSensorAdapter(model, data),
        controller=controller,
        attachment_synchronizer=controller,
        manipulator=CompositeManipulator(
            controller,
            controller,
            controller,
            controller,
            controller,
            environment,
        ),
        topology=topology,
        fleet_manager=fleet,
        traffic_manager=traffic,
        metrics=metrics,
        path_planner=path_planner,
        robot_registry=robots,
    )
    for spec in robot_specs:
        orchestrator.register_robot(spec.robot_id, spec.home)
    _initialize_inventory(inventory)
    return ApplicationRuntime(
        model,
        data,
        topology,
        inventory,
        reservations,
        metrics,
        environment,
        controller,
        scenario,
        fleet,
        traffic,
        robots,
        orchestrator,
    )


def _build_traffic_manager(
    topology: WarehouseTopology,
    path_planner: GridPathPlanner,
    robots: RobotRegistry,
) -> TrafficManager:
    traffic = TrafficManager(topology, path_planner, robots)
    traffic.set_strategy(
        TrafficPattern.REAR_END,
        RearEndStrategy(topology, path_planner),
    )
    traffic.set_strategy(
        TrafficPattern.HEAD_ON,
        HeadOnStrategy(topology, path_planner),
    )
    detour = SnapshotDetourStrategy(topology, path_planner)
    traffic.set_strategy(TrafficPattern.CROSSING, detour)
    traffic.set_strategy(TrafficPattern.STUCK_BETWEEN, detour)
    return traffic


def _initialize_inventory(inventory: InventoryWorld) -> None:
    physical = inventory.get_snapshot(SnapshotType.PHYSICAL)
    reserved = inventory.get_snapshot(SnapshotType.RESERVED)
    for x, y, tier, package_id in INITIAL_PACKAGES:
        physical.set_cell_occupancy(x, y, tier, package_id)
        reserved.set_cell_occupancy(x, y, tier, package_id)
