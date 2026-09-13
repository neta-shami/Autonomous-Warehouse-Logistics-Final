"""Autonomous robot context with atomic sensing, commit, and act phases."""

import math
from typing import List, Optional, Set, Tuple

from entities.drive_command import DriveCommand
from entities.enums import (
    CommandGateReason,
    FaultCode,
    FaultSeverity,
    FSMStatus,
    LocalizationStatus,
    ManipulatorAbortResult,
    TaskStatus,
    TaskType,
)
from entities.events import FaultResetEvent, WarehouseEvent
from entities.fleet_snapshot import FleetSnapshot
from entities.health import HealthCheckContext, HealthFinding
from entities.localization_assessment import LocalizationAssessment
from entities.robot_observation import RobotObservation
from entities.robot_state import RobotState
from entities.runtime_control import RobotTickAssessment, TickContext
from entities.sensor_frame import SensorFrame
from entities.task import Task
from entities.value_validation import require_finite
from entities.warehouse_topology import WarehouseTopology
from interfaces.i_drive_system import IDriveSystem
from interfaces.i_local_collision_avoidance import ILocalCollisionAvoidance
from interfaces.i_manipulator_system import IManipulatorSystem
from interfaces.i_path_planner import IPathPlanner
from use_cases.payload_presence_tracker import PayloadPresenceTracker
from use_cases.pose_fusion_service import PoseFusionService
from use_cases.robot_health_monitor import RobotHealthMonitor
from use_cases.robot_states import (
    FaultedState,
    IRobotState,
    NavigatingState,
    RelocalizingState,
)
from use_cases.task_strategies import (
    ITaskStrategy,
    ParkTaskStrategy,
    TransferTaskStrategy,
)
from use_cases.waypoint_controller import WaypointController


class RobotAgent:
    """Own one robot's beliefs, local decisions, and hardware command record."""

    WAYPOINT_TOLERANCE = 0.5
    FINAL_WAYPOINT_TOLERANCE = 0.1
    FOOTPRINT_RADIUS_M = 0.35

    def __init__(
        self,
        robot_state: RobotState,
        drive_system: IDriveSystem,
        manipulator_system: IManipulatorSystem,
        topology: WarehouseTopology,
        local_avoidance: ILocalCollisionAvoidance,
        path_planner: IPathPlanner,
        pose_fusion: PoseFusionService,
        health_monitor: RobotHealthMonitor,
        waypoint_controller: WaypointController,
        payload_tracker: PayloadPresenceTracker,
    ):
        self.state = robot_state
        self._drive_system = drive_system
        self.manipulator_system = manipulator_system
        self.topology = topology
        self.local_avoidance = local_avoidance
        self.path_planner = path_planner
        self.pose_fusion = pose_fusion
        self.health_monitor = health_monitor
        self.waypoint_controller = waypoint_controller
        self.payload_tracker = payload_tracker

        self.current_task: Optional[Task] = None
        self.current_leg_goal: Optional[Tuple[int, int]] = None
        self.path: List[Tuple[int, int]] = []
        self.path_index = 0
        self.event_bus: List[WarehouseEvent] = []
        self.task_strategy: Optional[ITaskStrategy] = None
        self.last_applied_drive_command = DriveCommand(0.0, 0.0)
        self._current_state: IRobotState = RelocalizingState()
        self.state_start_sim_time: Optional[float] = None
        self._relocalizing_due_to_loss = False
        self.latest_fleet_snapshot: Optional[FleetSnapshot] = None
        self.route_replan_required = False
        self._auto_clear_faults: Set[FaultCode] = set()

    @property
    def fsm_status(self) -> FSMStatus:
        return self._current_state.fsm_status

    @property
    def current_state(self) -> IRobotState:
        """Expose state identity for status display and focused tests."""
        return self._current_state

    def transition_to(
        self, state: IRobotState, sim_time: Optional[float] = None
    ) -> None:
        if not isinstance(state, IRobotState):
            raise TypeError("state must implement IRobotState")
        self._current_state = state
        self.state_start_sim_time = sim_time

    def assign_task(self, task: Task) -> None:
        """Assign only when a bounded pose makes execution safe."""
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if self.current_task is not None or self.state.is_busy:
            raise RuntimeError(
                f"Robot '{self.state.robot_id}' already has active work"
            )
        if self.state.localization_status not in (
            LocalizationStatus.TRUSTED,
            LocalizationStatus.DEGRADED,
        ) or self.state.pose_estimate is None:
            raise RuntimeError(
                f"Robot '{self.state.robot_id}' cannot accept work without a pose"
            )
        if self.state.active_faults:
            raise RuntimeError(
                f"Robot '{self.state.robot_id}' cannot accept work while faulted"
            )
        self.current_task = task
        task.assigned_robot_id = self.state.robot_id
        self.current_task.update_status(TaskStatus.IN_PROGRESS)
        self.state.is_busy = True
        self.state.expected_payload_id = task.payload_id
        current = (
            int(round(self.state.current_coords[0])),
            int(round(self.state.current_coords[1])),
        )
        if task.task_type in (
            TaskType.STORE,
            TaskType.RETRIEVE,
            TaskType.RELOCATE,
        ):
            if task.source_coords is None:
                raise ValueError("transfer task requires source coordinates")
            self.task_strategy = TransferTaskStrategy()
            self.current_leg_goal = self.topology.get_access_point(
                task.source_coords, current
            )
        elif task.task_type is TaskType.PARK:
            self.task_strategy = ParkTaskStrategy()
            self.current_leg_goal = self.topology.get_access_point(
                task.target_coords, current
            )
        self.transition_to(NavigatingState())

    def set_path(self, path: List[Tuple[int, int]]) -> None:
        self.path = list(path)
        self.path_index = 0
        if path:
            self.route_replan_required = False
        if self.fsm_status is FSMStatus.AVOIDING:
            self.transition_to(NavigatingState())

    def release_task_for_requeue(self, task: Task) -> None:
        """Detach only matching pre-pick work while preserving its lease."""
        if task is not self.current_task:
            raise ValueError("only the currently assigned task can be requeued")
        self.current_task = None
        self.task_strategy = None
        self.current_leg_goal = None
        self.path = []
        self.path_index = 0
        self.route_replan_required = False
        self.state.is_busy = False
        self.state.expected_payload_id = None
        task.assigned_robot_id = None

    def apply_drive(self, command: DriveCommand) -> None:
        """Apply and record exactly one already-bounded drive command."""
        if not isinstance(command, DriveCommand):
            raise TypeError("command must be a DriveCommand")
        if math.hypot(command.vx, command.vy) > (
            self.waypoint_controller.max_speed_mps + 1e-12
        ):
            raise ValueError("drive command exceeds configured maximum speed")
        self._drive_system.command_velocity(self.state.robot_id, command)
        self.last_applied_drive_command = command

    def stop_drive(self) -> None:
        """Stop hardware and record zero for prediction on every stop path."""
        self._drive_system.stop(self.state.robot_id)
        self.last_applied_drive_command = DriveCommand(0.0, 0.0)

    def assess(
        self,
        frame: Optional[SensorFrame],
        sim_time: float,
        dt: float,
    ) -> RobotTickAssessment:
        """Prepare all next state without mutating beliefs or commanding hardware."""
        require_finite("assessment sim_time", sim_time, minimum=0.0)
        require_finite("assessment dt", dt, minimum=0.0)
        if self.state_start_sim_time is None:
            self.state_start_sim_time = sim_time
        safe_frame, frame_fault = self._validate_frame(frame, sim_time)
        localization = self.pose_fusion.assess(
            safe_frame,
            self.state.pose_estimate,
            self.state.localization_status,
            self.last_applied_drive_command,
            dt,
        )
        if frame_fault and not localization.stop_required:
            localization = LocalizationAssessment(
                estimate=None,
                status=(
                    LocalizationStatus.UNINITIALIZED
                    if self.state.pose_estimate is None
                    else LocalizationStatus.LOST
                ),
                stop_required=True,
                divergence_m=localization.divergence_m,
                confirmation=localization.confirmation,
                events=localization.events,
            )

        payload = self.payload_tracker.assess(
            None if safe_frame is None else safe_frame.payload_range
        )
        if safe_frame is None:
            observation = RobotObservation(None, None, None, None, None)
        else:
            observation = RobotObservation(
                safe_frame.observed_chassis_velocity,
                payload.payload_sample,
                payload.payload_present,
                safe_frame.sim_time,
                safe_frame.sequence,
            )

        findings = list(
            self.health_monitor.evaluate(
                HealthCheckContext(
                    robot=self.state,
                    frame=safe_frame,
                    fsm_status=self.fsm_status,
                    state_duration_s=sim_time - self.state_start_sim_time,
                    last_applied_drive_command=self.last_applied_drive_command,
                    current_task_id=(
                        None if self.current_task is None else self.current_task.task_id
                    ),
                    task_phase=(
                        None
                        if self.current_task is None
                        else self.current_task.phase
                    ),
                    payload_present=payload.payload_present,
                    sim_time=sim_time,
                )
            )
        )
        if frame_fault:
            findings.append(
                HealthFinding(
                    FaultCode.SENSOR_INVALID,
                    FaultSeverity.STOP,
                    True,
                    False,
                    None,
                )
            )
        if (
            localization.estimate is not None
            and not self.topology.is_pose_feasible(
                localization.estimate.pose, self.FOOTPRINT_RADIUS_M
            )
        ):
            findings.append(
                HealthFinding(
                    FaultCode.POSITION_PHYSICALLY_INVALID,
                    FaultSeverity.STOP,
                    True,
                    True,
                    None,
                )
            )

        finding_tuple = tuple(findings)
        effective_faults = set(self.state.active_faults).difference(
            self._auto_clear_faults
        )
        effective_faults.update(finding.fault for finding in finding_tuple)
        events = localization.events + tuple(
            finding.event
            for finding in finding_tuple
            if finding.event is not None
            and finding.fault not in self.state.active_faults
        )
        return RobotTickAssessment(
            localization=localization,
            observation=observation,
            payload_presence=payload,
            health_findings=finding_tuple,
            stop_required=(
                localization.stop_required
                or any(finding.stop_required for finding in finding_tuple)
                or bool(effective_faults)
            ),
            events=events,
        )

    def commit(self, assessment: RobotTickAssessment) -> None:
        """Atomically publish an accepted assessment to this robot's state."""
        if not isinstance(assessment, RobotTickAssessment):
            raise TypeError("assessment must be RobotTickAssessment")
        self.pose_fusion.commit_confirmation(
            assessment.localization.confirmation
        )
        self.payload_tracker.commit(assessment.payload_presence)
        self.state.apply_localization(
            assessment.localization.estimate,
            assessment.localization.status,
        )
        observation = assessment.observation
        self.state.apply_observation(
            observation.observed_chassis_velocity,
            observation.payload_present,
            observation.sample_sim_time,
            observation.sequence,
        )
        active_faults = set(self.state.active_faults).difference(
            self._auto_clear_faults
        )
        active_faults.update(
            finding.fault for finding in assessment.health_findings
        )
        self.state.replace_faults(frozenset(active_faults))
        self._auto_clear_faults = {
            finding.fault
            for finding in assessment.health_findings
            if not finding.latched
        }
        self.event_bus.extend(assessment.events)

    def act(self, context: TickContext) -> None:
        """Act once against the fleet snapshot, or synchronously stop for a gate."""
        if not isinstance(context, TickContext):
            raise TypeError("context must be TickContext")
        self.latest_fleet_snapshot = context.fleet_snapshot
        if not context.commands_allowed:
            was_recovery_state = self.fsm_status in (
                FSMStatus.RELOCALIZING,
                FSMStatus.FAULTED,
            )
            if (
                context.gate_reasons
                == frozenset({CommandGateReason.FLEET_HOLD})
                and self.state_start_sim_time is not None
            ):
                self.state_start_sim_time += context.dt
            self.stop_drive()
            if self.fsm_status is FSMStatus.MANIPULATING:
                abort_result = self.manipulator_system.abort(
                    self.state.robot_id
                )
                if (
                    abort_result is ManipulatorAbortResult.RECONCILIATION_REQUIRED
                    and self.current_task is not None
                ):
                    self.current_task.update_status(
                        TaskStatus.RECOVERY_REQUIRED
                    )
                    self.state.activate_faults(FaultCode.MANIPULATOR_FAILURE)
                    self.transition_to(FaultedState())
                    return
            if CommandGateReason.LOCAL_SAFETY in context.gate_reasons:
                non_sensor_faults = self.state.active_faults.difference(
                    {FaultCode.SENSOR_INVALID}
                )
                if non_sensor_faults:
                    self.transition_to(FaultedState(), context.sim_time)
                elif self.fsm_status not in (
                    FSMStatus.RELOCALIZING,
                    FSMStatus.FAULTED,
                ):
                    self._relocalizing_due_to_loss = True
                    self.transition_to(RelocalizingState(), context.sim_time)
            if was_recovery_state and self.fsm_status in (
                FSMStatus.RELOCALIZING,
                FSMStatus.FAULTED,
            ):
                self._current_state.update(self, context)
            return
        self._current_state.update(self, context)

    def request_fault_reset(
        self, fault: FaultCode, context: HealthCheckContext
    ) -> bool:
        """Clear one fault only after its monitor validates supplied evidence."""
        if fault not in self.state.active_faults:
            return False
        if not self.health_monitor.can_clear(fault, context):
            return False
        self.state.clear_fault(fault)
        self._auto_clear_faults.discard(fault)
        self.event_bus.append(
            FaultResetEvent(context.sim_time, self.state.robot_id, fault)
        )
        if not self.state.active_faults and self.fsm_status is FSMStatus.FAULTED:
            self.transition_to(RelocalizingState(), context.sim_time)
        return True

    def _validate_frame(
        self, frame: Optional[SensorFrame], sim_time: float
    ) -> Tuple[Optional[SensorFrame], bool]:
        if frame is None or frame.robot_id != self.state.robot_id:
            return None, True
        if abs(frame.sim_time - sim_time) > 1e-9:
            return None, True
        previous_sequence = self.state.last_observation_sequence
        previous_time = self.state.last_observation_sim_time
        if previous_sequence is not None and frame.sequence <= previous_sequence:
            return None, True
        if previous_time is not None and frame.sim_time <= previous_time:
            return None, True
        return frame, False
