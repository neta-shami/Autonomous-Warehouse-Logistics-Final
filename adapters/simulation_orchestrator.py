"""Atomic MuJoCo physics/sensing/domain orchestration."""

import logging
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Set, Tuple

from entities.enums import (
    CommandGateReason,
    FaultCode,
    LocalizationStatus,
    TaskType,
)
from entities.events import (
    CollisionStallEvent,
    FaultResetEvent,
    PackageDroppedEvent,
    PackagePickedEvent,
    PayloadMismatchEvent,
    RelocalizationFailedEvent,
    RelocalizedEvent,
    RobotFaultedEvent,
    RobotIdleEvent,
    TaskCompletedEvent,
    TaskReadyEvent,
    WarehouseEvent,
)
from entities.fleet_snapshot import FleetRobotView, FleetSnapshot
from entities.health import HealthCheckContext
from entities.robot_state import RobotState
from entities.runtime_control import RobotTickAssessment, TickContext
from entities.sensor_frame import SensorFrame
from entities.warehouse_topology import WarehouseTopology
from interfaces.i_attachment_synchronizer import IAttachmentSynchronizer
from interfaces.i_drive_system import IDriveSystem
from interfaces.i_environment import IEnvironment
from interfaces.i_manipulator_system import IManipulatorSystem
from interfaces.i_robot_registry import ICoordinatedRobot, IRobotRegistry
from interfaces.i_sensor_adapter import ISensorAdapter
from interfaces.sensor_read_error import SensorReadError
from use_cases.continuous_wall_range_localizer import (
    ContinuousWallRangeLocalizer,
)
from use_cases.event_dispatcher import EventDispatcher
from use_cases.fleet_manager import FleetManager
from use_cases.grid_path_planner import GridPathPlanner
from use_cases.holonomic_command_motion_model import HolonomicCommandMotionModel
from use_cases.local_collision_avoidance import LocalCollisionAvoidance
from use_cases.metrics_collector import MetricsCollector
from use_cases.payload_check import PayloadCheck
from use_cases.payload_presence_tracker import PayloadPresenceTracker
from use_cases.pose_fusion_service import PoseFusionService
from use_cases.robot_agent import RobotAgent
from use_cases.robot_health_monitor import RobotHealthMonitor
from use_cases.stall_check import StallCheck
from use_cases.state_timeout_check import StateTimeoutCheck
from use_cases.traffic_manager import TrafficManager
from use_cases.warehouse_scenario_manager import WarehouseScenarioManager
from use_cases.waypoint_controller import WaypointController

LOGGER = logging.getLogger(__name__)


class SimulationOrchestrator:
    """Run read-all/assess-all/commit-all/act-all logic ticks."""

    PHYSICS_STEPS_PER_LOGIC_TICK = 10
    MAX_DRIVE_SPEED_MPS = 1.0
    MAX_EVENTS_PER_LOGIC_TICK = 10_000

    def __init__(
        self,
        environment: IEnvironment,
        scenario_manager: WarehouseScenarioManager,
        sensor_adapter: ISensorAdapter,
        controller: IDriveSystem,
        attachment_synchronizer: IAttachmentSynchronizer,
        manipulator: IManipulatorSystem,
        topology: WarehouseTopology,
        fleet_manager: FleetManager,
        traffic_manager: TrafficManager,
        metrics: MetricsCollector,
        path_planner: GridPathPlanner,
        robot_registry: IRobotRegistry,
        physics_steps_per_logic_tick: int = PHYSICS_STEPS_PER_LOGIC_TICK,
    ):
        if physics_steps_per_logic_tick < 1:
            raise ValueError(
                "physics_steps_per_logic_tick must be at least 1, "
                f"got {physics_steps_per_logic_tick}"
            )
        self.physics_steps_per_logic_tick = physics_steps_per_logic_tick
        self.environment = environment
        self.scenario_manager = scenario_manager
        self.sensor_adapter = sensor_adapter
        self.controller = controller
        self.attachment_synchronizer = attachment_synchronizer
        self.manipulator = manipulator
        self.topology = topology
        self.fleet_manager = fleet_manager
        self.traffic_manager = traffic_manager
        self.metrics = metrics
        self.path_planner = path_planner
        self.robot_registry = robot_registry
        self._snapshot_version = 0
        self._last_logic_sim_time: Optional[float] = None
        self._latest_frames: Dict[str, Optional[SensorFrame]] = {}

        self.event_dispatcher = EventDispatcher()
        self.fleet_manager.reservation_service.set_event_dispatcher(
            self.event_dispatcher
        )
        self._setup_event_subscriptions()

    @property
    def agents(self) -> Mapping[str, ICoordinatedRobot]:
        """Expose a read-only robot mapping for status views and tests."""
        return MappingProxyType(dict(self.robot_registry.items()))

    @agents.setter
    def agents(self, robots: Mapping[str, ICoordinatedRobot]) -> None:
        """Replace robots while constructing isolated orchestrator fixtures."""
        self.robot_registry.replace(robots.items())

    def _robot(self, robot_id: str) -> ICoordinatedRobot:
        robot = self.robot_registry.get(robot_id)
        if robot is None:
            raise KeyError(f"Unknown robot '{robot_id}'")
        return robot

    def _setup_event_subscriptions(self) -> None:
        self.event_dispatcher.subscribe(
            CollisionStallEvent, self.traffic_manager.handle_stall_event
        )
        self.event_dispatcher.subscribe(
            CollisionStallEvent, self.metrics.handle_collision_stall
        )
        self.event_dispatcher.subscribe(
            PackagePickedEvent,
            self.fleet_manager.reservation_service.handle_package_picked,
        )
        self.event_dispatcher.subscribe(
            PackageDroppedEvent,
            self.fleet_manager.reservation_service.handle_package_dropped,
        )
        self.event_dispatcher.subscribe(
            TaskCompletedEvent, self.scenario_manager.handle_task_completed
        )
        self.event_dispatcher.subscribe(
            TaskCompletedEvent,
            self.fleet_manager.reservation_service.handle_task_completed,
        )
        self.event_dispatcher.subscribe(
            TaskCompletedEvent, self.metrics.handle_task_completed
        )
        self.event_dispatcher.subscribe(
            RobotIdleEvent, self.fleet_manager.handle_robot_idle
        )
        self.event_dispatcher.subscribe(
            TaskReadyEvent, self.fleet_manager.handle_task_ready
        )
        self.event_dispatcher.subscribe(
            RelocalizedEvent, self.fleet_manager.handle_relocalized
        )
        self.event_dispatcher.subscribe(
            RobotFaultedEvent, self.fleet_manager.handle_robot_faulted
        )
        self.event_dispatcher.subscribe(
            PayloadMismatchEvent, self.fleet_manager.handle_payload_mismatch
        )
        self.event_dispatcher.subscribe(
            RelocalizedEvent, self.metrics.handle_relocalized
        )
        self.event_dispatcher.subscribe(
            RelocalizationFailedEvent,
            self.metrics.handle_relocalization_failed,
        )
        self.event_dispatcher.subscribe(
            FaultResetEvent, self.fleet_manager.handle_fault_reset
        )

    def register_robot(
        self, robot_id: str, home_base_coords: tuple[float, float]
    ) -> None:
        """Register an unlocalized robot; home coordinates are not a pose fix."""
        if self.robot_registry.get(robot_id) is not None:
            raise ValueError(f"Robot '{robot_id}' is already registered")
        home = tuple(int(value) for value in home_base_coords)
        if len(home) != 2 or any(
            float(original) != converted
            for original, converted in zip(home_base_coords, home)
        ):
            raise ValueError("home_base_coords must contain two integer coordinates")
        state = RobotState(
            robot_id=robot_id,
            pose_estimate=None,
            observed_chassis_velocity=None,
            expected_payload_id=None,
            payload_present=None,
            is_busy=False,
            home_base_coords=home,
        )
        agent = RobotAgent(
            robot_state=state,
            drive_system=self.controller,
            manipulator_system=self.manipulator,
            topology=self.topology,
            local_avoidance=LocalCollisionAvoidance(),
            path_planner=self.path_planner,
            pose_fusion=PoseFusionService(
                HolonomicCommandMotionModel(),
                ContinuousWallRangeLocalizer(
                    pair_residual_tolerance_m=0.05,
                    measurement_error_bound_m=0.01,
                ),
                self.topology,
                max_speed_mps=self.MAX_DRIVE_SPEED_MPS,
            ),
            health_monitor=RobotHealthMonitor(
                [StallCheck(), PayloadCheck(), StateTimeoutCheck()]
            ),
            waypoint_controller=WaypointController(self.MAX_DRIVE_SPEED_MPS),
            payload_tracker=PayloadPresenceTracker(),
        )
        self.robot_registry.register(robot_id, agent)

    def step(self) -> None:
        """Advance physics, then execute one deterministic atomic logic tick."""
        robot_ids = self.robot_registry.robot_ids()
        collision_pairs = self._advance_physics_phase(robot_ids)
        sim_time, dt = self._capture_tick_time_phase()
        self._sample_pre_control_metrics_phase(robot_ids, sim_time, collision_pairs)
        frames = self._read_sensor_frames_phase(robot_ids)
        assessments = self._assess_and_commit_phase(robot_ids, frames, sim_time, dt)
        snapshot, fleet_hold = self._coordinate_fleet_phase(sim_time)
        self._act_phase(
            robot_ids,
            frames,
            assessments,
            snapshot,
            sim_time,
            dt,
            fleet_hold,
        )
        self._publish_and_observe_phase(robot_ids, sim_time, dt)

    def _advance_physics_phase(
        self, robot_ids: Tuple[str, ...]
    ) -> Set[Tuple[str, str]]:
        """Advance the physical world and collect collisions for this tick."""
        collision_pairs: Set[Tuple[str, str]] = set()
        for _ in range(self.physics_steps_per_logic_tick):
            self.environment.step()
            self.attachment_synchronizer.sync_attachments()
            collision_pairs.update(self.environment.get_robot_collisions(robot_ids))
        return collision_pairs

    def _capture_tick_time_phase(self) -> Tuple[float, float]:
        """Read and validate the single timestamp shared by the logic tick."""
        sim_time = self.environment.get_time()
        dt = (
            0.0
            if self._last_logic_sim_time is None
            else sim_time - self._last_logic_sim_time
        )
        if dt < 0.0:
            raise ValueError("simulation time moved backwards")
        self._last_logic_sim_time = sim_time
        return sim_time, dt

    def _sample_pre_control_metrics_phase(
        self,
        robot_ids: Tuple[str, ...],
        sim_time: float,
        collision_pairs: Set[Tuple[str, str]],
    ) -> None:
        """Measure physical outcomes produced by the preceding commands."""
        self.metrics.record_robot_collisions(collision_pairs)
        active_transfer_tasks = {
            robot_id: agent.current_task.task_id
            for robot_id, agent in self.robot_registry.items()
            if agent.current_task is not None
            and agent.current_task.task_type is not TaskType.PARK
        }
        self.metrics.sample_physical_motion(robot_ids, sim_time, active_transfer_tasks)

    def _read_sensor_frames_phase(
        self, robot_ids: Tuple[str, ...]
    ) -> Dict[str, Optional[SensorFrame]]:
        """Capture at most one immutable sensor frame per robot."""
        frames: Dict[str, Optional[SensorFrame]] = {}
        for robot_id in robot_ids:
            try:
                frames[robot_id] = self.sensor_adapter.read_frame(robot_id)
            except SensorReadError as exc:
                LOGGER.warning(
                    "Sensor frame unavailable for robot %s: %s",
                    robot_id,
                    exc,
                )
                frames[robot_id] = None
        self._latest_frames = dict(frames)
        return frames

    def _assess_and_commit_phase(
        self,
        robot_ids: Tuple[str, ...],
        frames: Dict[str, Optional[SensorFrame]],
        sim_time: float,
        dt: float,
    ) -> Dict[str, RobotTickAssessment]:
        """Assess every robot before atomically committing any proposal."""
        assessments = {
            robot_id: self._robot(robot_id).assess(
                frames[robot_id], sim_time, dt
            )
            for robot_id in robot_ids
        }
        for robot_id in robot_ids:
            self._robot(robot_id).commit(assessments[robot_id])
        return assessments

    def _coordinate_fleet_phase(self, sim_time: float) -> Tuple[FleetSnapshot, bool]:
        """Publish one shared snapshot and calculate the fleet-wide hold."""
        snapshot = self._build_snapshot(sim_time)
        self.fleet_manager.update_snapshot(snapshot)
        self.traffic_manager.update_snapshot(snapshot)
        fleet_hold = any(
            view.pose_estimate is None
            or view.localization_status
            in (LocalizationStatus.UNINITIALIZED, LocalizationStatus.LOST)
            for view in snapshot.robots
        )
        return snapshot, fleet_hold

    def _act_phase(
        self,
        robot_ids: Tuple[str, ...],
        frames: Dict[str, Optional[SensorFrame]],
        assessments: Dict[str, RobotTickAssessment],
        snapshot: FleetSnapshot,
        sim_time: float,
        dt: float,
        fleet_hold: bool,
    ) -> None:
        """Apply safety-gated commands using the same committed snapshot."""
        for robot_id in robot_ids:
            reasons = set()
            if assessments[robot_id].stop_required:
                reasons.add(CommandGateReason.LOCAL_SAFETY)
            if fleet_hold:
                reasons.add(CommandGateReason.FLEET_HOLD)
            if robot_id in self.fleet_manager.command_holds:
                reasons.add(CommandGateReason.FLEET_HOLD)
            effective_frame = (
                frames[robot_id]
                if assessments[robot_id].observation.sequence is not None
                else None
            )
            robot = self._robot(robot_id)
            robot.act(
                TickContext(
                    frame=effective_frame,
                    fleet_snapshot=snapshot,
                    sim_time=sim_time,
                    dt=dt,
                    gate_reasons=frozenset(reasons),
                )
            )
            self.metrics.record_command_result(
                sim_time,
                unsafe=bool(reasons),
                command=robot.last_applied_drive_command,
            )

    def _publish_and_observe_phase(
        self,
        robot_ids: Tuple[str, ...],
        sim_time: float,
        dt: float,
    ) -> None:
        """Publish nested events, then sample post-control system health."""
        # No callback can alter the snapshot before every robot has acted.
        self.metrics.sample_task_progress(
            self.robot_registry.values(), sim_time, dt
        )
        self._dispatch_agent_events(robot_ids)

        self.metrics.sample_localization_error(
            (robot.state for robot in self.robot_registry.values()), sim_time
        )
        self.scenario_manager.tick(sim_time)
        active_task_ids = {
            agent.current_task.task_id
            for agent in self.robot_registry.values()
            if agent.current_task is not None
        }
        active_task_ids.update(
            task.task_id
            for task in (
                self.fleet_manager.reservation_service.ready_queue
                + self.fleet_manager.reservation_service.blocked_queue
            )
        )
        self.metrics.sample_orphan_leases(
            self.fleet_manager.reservation_service.inventory_world,
            active_task_ids,
        )

    def _dispatch_agent_events(self, robot_ids: tuple[str, ...]) -> None:
        """Drain nested same-tick events without erasing handler output."""
        dispatched = 0
        while True:
            made_progress = False
            for robot_id in robot_ids:
                event_bus = self._robot(robot_id).event_bus
                if not event_bus:
                    continue
                event = event_bus.pop(0)
                made_progress = True
                dispatched += 1
                if dispatched > self.MAX_EVENTS_PER_LOGIC_TICK:
                    raise RuntimeError(
                        "Event dispatch exceeded the per-tick safety limit"
                    )
                if isinstance(event, WarehouseEvent):
                    LOGGER.info(
                        "Event at %.2f id=%s type=%s",
                        event.sim_time,
                        event.event_id[:8],
                        type(event).__name__,
                    )
                self.event_dispatcher.dispatch(event)
            if not made_progress:
                return

    def _build_snapshot(self, sim_time: float) -> FleetSnapshot:
        self._snapshot_version += 1
        views = tuple(
            FleetRobotView(
                robot_id=robot_id,
                pose_estimate=agent.state.pose_estimate,
                observed_chassis_velocity=agent.state.observed_chassis_velocity,
                footprint_radius_m=agent.FOOTPRINT_RADIUS_M,
                active_path=tuple(agent.path[agent.path_index :]),
                localization_status=agent.state.localization_status,
                active_faults=agent.state.active_faults,
                fsm_status=agent.fsm_status,
            )
            for robot_id, agent in self.robot_registry.items()
        )
        return FleetSnapshot(self._snapshot_version, sim_time, views)

    def spawn_package(
        self,
        package_id: str,
        target_shelf: tuple[int, int],
        target_tier: int,
    ) -> bool:
        self.topology.validate_shelf_slot(
            target_shelf, target_tier, field_name="package spawn target"
        )
        sim_time = self.environment.get_time()
        return self.scenario_manager.spawn_package(
            package_id, target_shelf, target_tier, sim_time
        )

    def request_fault_reset(
        self,
        robot_id: str,
        fault: FaultCode,
        *,
        inventory_reconciled: bool = False,
        drive_diagnostic_passed: bool = False,
        drive_cause_removed: bool = False,
    ) -> bool:
        """Submit explicit recovery evidence through the normal audited event flow."""
        if robot_id not in self.agents:
            raise KeyError(f"Unknown robot '{robot_id}'")
        if not isinstance(fault, FaultCode):
            raise TypeError("fault must be FaultCode")
        agent = self._robot(robot_id)
        sim_time = self.environment.get_time()
        estimate = agent.state.pose_estimate
        position_feasible = bool(
            estimate is not None
            and self.topology.is_pose_feasible(estimate.pose, agent.FOOTPRINT_RADIUS_M)
        )
        state_duration = (
            0.0
            if agent.state_start_sim_time is None
            else max(0.0, sim_time - agent.state_start_sim_time)
        )
        return agent.request_fault_reset(
            fault,
            HealthCheckContext(
                robot=agent.state,
                frame=self._latest_frames.get(robot_id),
                fsm_status=agent.fsm_status,
                state_duration_s=state_duration,
                last_applied_drive_command=agent.last_applied_drive_command,
                current_task_id=(
                    None if agent.current_task is None else agent.current_task.task_id
                ),
                task_phase=(
                    None if agent.current_task is None else agent.current_task.phase
                ),
                payload_present=agent.state.payload_present,
                sim_time=sim_time,
                drive_diagnostic_passed=drive_diagnostic_passed,
                drive_cause_removed=drive_cause_removed,
                inventory_reconciled=inventory_reconciled,
                position_feasible=position_feasible,
            ),
        )
