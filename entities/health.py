"""Framework-free inputs and outputs for robot health checks."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from entities.drive_command import DriveCommand
from entities.enums import FaultCode, FaultSeverity, FSMStatus, TaskExecutionPhase
from entities.sensor_frame import SensorFrame
from entities.value_validation import require_finite

if TYPE_CHECKING:
    from entities.events import WarehouseEvent
    from entities.robot_state import RobotState


@dataclass(frozen=True)
class HealthCheckContext:
    """Committed robot state and same-frame evidence available to one check."""

    robot: "RobotState"
    frame: Optional[SensorFrame]
    fsm_status: FSMStatus
    state_duration_s: float
    last_applied_drive_command: DriveCommand
    current_task_id: Optional[str]
    task_phase: Optional[TaskExecutionPhase]
    payload_present: Optional[bool] = None
    sim_time: float = 0.0
    drive_diagnostic_passed: bool = False
    drive_cause_removed: bool = False
    inventory_reconciled: bool = False
    position_feasible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.fsm_status, FSMStatus):
            raise TypeError("fsm_status must be FSMStatus")
        require_finite("state_duration_s", self.state_duration_s, minimum=0.0)
        if not isinstance(self.last_applied_drive_command, DriveCommand):
            raise TypeError("last_applied_drive_command must be DriveCommand")
        if self.current_task_id is not None and (
            not isinstance(self.current_task_id, str) or not self.current_task_id
        ):
            raise ValueError("current_task_id must be a non-empty string or None")
        if self.task_phase is not None and not isinstance(
            self.task_phase, TaskExecutionPhase
        ):
            raise TypeError("task_phase must be TaskExecutionPhase or None")
        if self.payload_present is not None and not isinstance(
            self.payload_present, bool
        ):
            raise TypeError("payload_present must be bool or None")
        require_finite("health sim_time", self.sim_time, minimum=0.0)
        boolean_fields = (
            ("drive_diagnostic_passed", self.drive_diagnostic_passed),
            ("drive_cause_removed", self.drive_cause_removed),
            ("inventory_reconciled", self.inventory_reconciled),
            ("position_feasible", self.position_feasible),
        )
        for name, value in boolean_fields:
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")


@dataclass(frozen=True)
class HealthFinding:
    """One independently reported operational fault."""

    fault: FaultCode
    severity: FaultSeverity
    stop_required: bool
    latched: bool
    event: Optional["WarehouseEvent"]

    def __post_init__(self) -> None:
        if not isinstance(self.fault, FaultCode):
            raise TypeError("fault must be FaultCode")
        if not isinstance(self.severity, FaultSeverity):
            raise TypeError("severity must be FaultSeverity")
        if not isinstance(self.stop_required, bool):
            raise TypeError("stop_required must be bool")
        if not isinstance(self.latched, bool):
            raise TypeError("latched must be bool")
