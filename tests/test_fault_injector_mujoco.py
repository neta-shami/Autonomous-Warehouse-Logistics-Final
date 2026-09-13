"""Real-physics displacement detection and recovery integration proof."""

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from adapters.fault_injector import FaultInjector
from adapters.mujoco_controller import MuJoCoController
from adapters.mujoco_environment_adapter import MuJoCoEnvironmentAdapter
from adapters.mujoco_ground_truth_probe import MuJoCoGroundTruthProbe
from adapters.sensor_adapter import MuJoCoSensorAdapter
from adapters.simulation_loader import SimulationLoader
from adapters.simulation_orchestrator import SimulationOrchestrator
from entities.enums import (
    FaultCode,
    FSMStatus,
    LocalizationStatus,
    TaskExecutionPhase,
)
from entities.inventory_world import InventoryWorld
from entities.pose import Pose
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
from use_cases.warehouse_scenario_manager import WarehouseScenarioManager


def _single_robot_simulation():
    model, data, topology = SimulationLoader(
        str(Path(__file__).parents[1] / "warehouse.xml")
    ).load()
    environment = MuJoCoEnvironmentAdapter(model, data)
    metrics = MetricsCollector(MuJoCoGroundTruthProbe(model, data))
    sensor = FaultInjector(
        MuJoCoSensorAdapter(model, data), metrics, random_seed=20260906
    )
    controller = MuJoCoController(model, data)
    manipulator = CompositeManipulator(
        controller,
        controller,
        controller,
        controller,
        controller,
        environment,
    )
    reservations = InventoryReservationService(InventoryWorld(), topology)
    planner = GridPathPlanner(AStarPathfindingStrategy())
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
        sensor,
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
    return model, data, environment, metrics, sensor, controller, orchestrator


def test_stationary_teleport_stops_same_tick_then_recovers_without_fake_distance():
    (
        _model,
        _data,
        environment,
        metrics,
        sensor,
        _controller,
        orchestrator,
    ) = _single_robot_simulation()
    for _ in range(3):
        orchestrator.step()
    agent = orchestrator.agents["r1"]
    assert agent.fsm_status is FSMStatus.IDLE

    sensor.teleport_robot(
        "r1", Pose(5.0, 5.0), environment.get_time(), environment.teleport_robot
    )
    orchestrator.step()

    assert agent.state.localization_status is LocalizationStatus.LOST
    assert agent.fsm_status is FSMStatus.RELOCALIZING
    assert agent.last_applied_drive_command.vx == 0.0
    assert agent.last_applied_drive_command.vy == 0.0
    assert metrics.unsafe_command_batches == 0

    for _ in range(2):
        orchestrator.step()

    assert agent.state.localization_status is LocalizationStatus.TRUSTED
    assert agent.state.pose_estimate.pose.distance_to(Pose(5.0, 5.0)) < 0.05
    assert metrics.automatic_recovery_rate == 1.0
    assert metrics.false_relocations == 0
    assert metrics.total_distance_driven < 0.1


def test_navigating_teleport_preserves_task_and_replans_from_physical_pose():
    """Exercise kidnapped-robot recovery while real MuJoCo motion is active."""
    (
        _model,
        _data,
        environment,
        metrics,
        sensor,
        _controller,
        orchestrator,
    ) = _single_robot_simulation()
    assert orchestrator.spawn_package("p1", (8, 2), 1)
    agent = orchestrator.agents["r1"]
    for _ in range(500):
        orchestrator.step()
        if agent.fsm_status is FSMStatus.NAVIGATING and agent.current_task:
            break
    else:
        pytest.fail("robot never started its physical navigation leg")
    task = agent.current_task

    sensor.teleport_robot(
        "r1", Pose(5.0, 8.0), environment.get_time(), environment.teleport_robot
    )
    orchestrator.step()

    assert agent.fsm_status is FSMStatus.RELOCALIZING
    assert agent.current_task is task
    assert agent.last_applied_drive_command.vx == 0.0
    assert agent.last_applied_drive_command.vy == 0.0
    for _ in range(3):
        orchestrator.step()
    assert agent.state.localization_status is LocalizationStatus.TRUSTED
    assert agent.current_task is task
    assert agent.state.pose_estimate.pose.distance_to(Pose(5.0, 8.0)) < 0.05
    assert metrics.automatic_recovery_rate == 1.0


def test_loaded_teleport_preserves_payload_lease_and_completes_transfer():
    """Exercise recovery with an attached package in the real physics model."""
    (
        _model,
        _data,
        environment,
        metrics,
        sensor,
        controller,
        orchestrator,
    ) = _single_robot_simulation()
    assert orchestrator.spawn_package("p1", (8, 2), 1)
    agent = orchestrator.agents["r1"]
    for _ in range(2_500):
        orchestrator.step()
        task = agent.current_task
        if (
            task is not None
            and task.phase is TaskExecutionPhase.TO_TARGET
            and agent.state.payload_present is True
        ):
            break
    else:
        pytest.fail("robot never carried the physical package")
    task = agent.current_task
    assert controller.attached_packages.get("r1") == "package_p1_free"

    sensor.teleport_robot(
        "r1", Pose(5.0, 5.0), environment.get_time(), environment.teleport_robot
    )
    orchestrator.step()

    assert agent.fsm_status is FSMStatus.RELOCALIZING
    assert agent.current_task is task
    assert controller.attached_packages.get("r1") == "package_p1_free"
    for _ in range(3):
        orchestrator.step()
    assert agent.state.localization_status is LocalizationStatus.TRUSTED
    assert agent.current_task is task

    for _ in range(2_500):
        orchestrator.step()
        if metrics.total_throughput == 1:
            break
    assert metrics.total_throughput == 1
    assert metrics.total_failed_tasks == 0
    assert controller.attached_packages == {}
    assert metrics.automatic_recovery_rate == 1.0
    assert metrics.unsafe_command_batches == 0
    assert metrics.orphan_lease_count == 0


def test_injected_physical_drive_stall_is_detected_without_losing_pose():
    (
        _model,
        _data,
        environment,
        _metrics,
        sensor,
        controller,
        orchestrator,
    ) = _single_robot_simulation()
    assert orchestrator.spawn_package("p1", (2, 2), 0)
    for _ in range(3):
        orchestrator.step()
    agent = orchestrator.agents["r1"]
    for _ in range(500):
        orchestrator.step()
        command = agent.last_applied_drive_command
        if agent.fsm_status is FSMStatus.NAVIGATING and (
            abs(command.vx) + abs(command.vy) > 0.1
        ):
            break
    else:
        pytest.fail("robot never produced a navigation command")

    sensor.inject_drive_stall("r1", True, controller.set_drive_stalled)
    for _ in range(100):
        orchestrator.step()
        if FaultCode.DRIVE_STALL in agent.state.active_faults:
            break
    sensor.inject_drive_stall("r1", False, controller.set_drive_stalled)

    assert FaultCode.DRIVE_STALL in agent.state.active_faults
    assert agent.state.localization_status is LocalizationStatus.TRUSTED
    assert environment.get_time() < 3.0


def test_stolen_attached_payload_stays_removed_after_attachment_sync():
    model, data, _ = SimulationLoader(
        str(Path(__file__).parents[1] / "warehouse.xml")
    ).load()
    environment = MuJoCoEnvironmentAdapter(model, data)
    controller = MuJoCoController(model, data)
    grip_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "robot_r1_grip_site"
    )
    grip_position = data.site_xpos[grip_site]
    assert environment.spawn_package_physically(
        "p1", float(grip_position[0]), float(grip_position[1]),
        float(grip_position[2])
    )
    mujoco.mj_forward(model, data)
    controller.grip("r1", "p1", engage=True)
    assert controller.attached_packages["r1"] == "package_p1_free"

    FaultInjector.steal_payload(
        "p1", controller.detach_payload, environment.remove_payload
    )
    controller.sync_attachments()

    package_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "package_p1_free"
    )
    address = model.jnt_qposadr[package_joint]
    assert "r1" not in controller.attached_packages
    assert tuple(data.qpos[address:address + 3]) == pytest.approx(
        (999.0, 999.0, 0.04)
    )
