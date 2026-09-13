import unittest

from adapters.mujoco_controller import MuJoCoController
from adapters.mujoco_environment_adapter import MuJoCoEnvironmentAdapter
from adapters.sensor_adapter import MuJoCoSensorAdapter
from adapters.simulation_loader import SimulationLoader
from adapters.simulation_orchestrator import SimulationOrchestrator
from entities.enums import FSMStatus, LocalizationStatus, TrafficPattern
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


class TestOrchestratorIntegration(unittest.TestCase):
    def test_minimal_simulation_loop(self):
        """
        Runs a 10 logic tick simulation loop (which is 10 * 50 physics steps)
        to ensure all dependencies are correctly wired and the loop doesn't crash.
        """
        inventory_world = InventoryWorld()
        strategy = AStarPathfindingStrategy()
        path_planner = GridPathPlanner(strategy)
        metrics = MetricsCollector()

        loader = SimulationLoader("warehouse.xml")
        model, data, topology = loader.load()
        reservation_service = InventoryReservationService(
            inventory_world, topology
        )

        environment = MuJoCoEnvironmentAdapter(model, data)
        sensor_adapter = MuJoCoSensorAdapter(model, data)
        controller = MuJoCoController(model, data)
        manipulator = CompositeManipulator(
            controller,
            controller,
            controller,
            controller,
            controller,
            environment,
        )

        scenario_manager = WarehouseScenarioManager(environment, reservation_service)

        allocation_strategy = GreedyTaskAllocationStrategy()
        robot_registry = RobotRegistry()
        fleet_manager = FleetManager(
            reservation_service,
            path_planner,
            topology,
            allocation_strategy,
            RouteBaselineEstimator(path_planner, topology),
            robot_registry,
        )

        traffic_manager = TrafficManager(topology, path_planner, robot_registry)
        traffic_manager.set_strategy(TrafficPattern.REAR_END, RearEndStrategy(topology, path_planner))
        traffic_manager.set_strategy(TrafficPattern.HEAD_ON, HeadOnStrategy(topology, path_planner))
        detour_strategy = SnapshotDetourStrategy(topology, path_planner)
        traffic_manager.set_strategy(TrafficPattern.CROSSING, detour_strategy)
        traffic_manager.set_strategy(TrafficPattern.STUCK_BETWEEN, detour_strategy)

        orchestrator = SimulationOrchestrator(
            environment=environment,
            scenario_manager=scenario_manager,
            sensor_adapter=sensor_adapter,
            controller=controller,
            attachment_synchronizer=controller,
            manipulator=manipulator,
            topology=topology,
            fleet_manager=fleet_manager,
            traffic_manager=traffic_manager,
            metrics=metrics,
            path_planner=path_planner,
            robot_registry=robot_registry,
        )

        orchestrator.register_robot("r1", (1.0, 1.0))
        agent = orchestrator.agents["r1"]
        self.assertIsNone(agent.state.pose_estimate)
        self.assertIs(agent.state.localization_status, LocalizationStatus.UNINITIALIZED)
        self.assertIs(agent.fsm_status, FSMStatus.RELOCALIZING)

        # Home configuration is intentionally wrong: sensors must establish
        # the XML body's physical start at (1, 9), not trust the supplied home.
        for _ in range(3):
            orchestrator.step()
        self.assertIs(agent.state.localization_status, LocalizationStatus.TRUSTED)
        self.assertAlmostEqual(agent.state.pose_estimate.pose.x, 1.0, places=1)
        self.assertAlmostEqual(agent.state.pose_estimate.pose.y, 9.0, places=1)
        self.assertIs(agent.fsm_status, FSMStatus.IDLE)

        orchestrator.spawn_package("p1", (2, 2), 1)

        # Continue the integration run after initialization.
        for _ in range(497):
            orchestrator.step()

        self.assertGreater(environment.get_time(), 0.0)

if __name__ == '__main__':
    unittest.main()
