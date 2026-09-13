import unittest

from adapters.mujoco_controller import MuJoCoController
from adapters.mujoco_environment_adapter import MuJoCoEnvironmentAdapter
from adapters.sensor_adapter import MuJoCoSensorAdapter
from adapters.simulation_loader import SimulationLoader
from adapters.simulation_orchestrator import SimulationOrchestrator
from entities.enums import FSMStatus, TrafficPattern
from entities.inventory_world import InventoryWorld
from entities.task import Task
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


class TestRobotMovement(unittest.TestCase):
    def setUp(self):
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

        self.orchestrator = SimulationOrchestrator(
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

    def test_robot_self_movement_no_orchestrator_intervention(self):
        """
        Verifies that a robot moves to its target solely through its inner FSM update tick,
        without the orchestrator forcing the movement.
        """
        # Home configuration is not localization evidence. The robot first
        # confirms its physical XML start pose at (1, 9) from wall sensors.
        self.orchestrator.register_robot("r1", (1.0, 9.0))
        agent = self.orchestrator.agents["r1"]
        for _ in range(4):
            self.orchestrator.step()

        # Manually assign a task to force it to navigate along the parking row.
        task = Task.park("test_task", (5, 9))
        agent.assign_task(task)
        agent.set_path([(2, 9), (3, 9), (4, 9), (5, 9)])

        self.assertEqual(agent.fsm_status, FSMStatus.NAVIGATING)

        # Initial position
        start_x = agent.state.current_coords[0]

        # Observe movement before the synthetic task reaches manipulation.
        for _ in range(50):
            self.orchestrator.step()

        # The robot should have moved towards the target because `agent.update_tick()`
        # issued a bounded velocity command through the robot-owned controller.
        current_x = agent.state.current_coords[0]

        # We expect x to have increased significantly from 1.0 towards 5.0
        self.assertGreater(current_x, start_x + 0.5)

if __name__ == '__main__':
    unittest.main()
