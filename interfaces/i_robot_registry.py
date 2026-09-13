"""Narrow registry port used by fleet and traffic coordination."""

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Protocol, Tuple

from entities.drive_command import DriveCommand
from entities.enums import FaultCode, FSMStatus
from entities.events import WarehouseEvent
from entities.health import HealthCheckContext
from entities.robot_state import RobotState
from entities.runtime_control import RobotTickAssessment, TickContext
from entities.sensor_frame import SensorFrame
from entities.task import Task


class ICoordinatedRobot(Protocol):
    """Capabilities required by orchestration, fleet, and traffic services."""

    FOOTPRINT_RADIUS_M: float
    WAYPOINT_TOLERANCE: float
    FINAL_WAYPOINT_TOLERANCE: float
    state: RobotState
    current_task: Optional[Task]
    current_leg_goal: Optional[Tuple[int, int]]
    path: List[Tuple[int, int]]
    path_index: int
    route_replan_required: bool
    last_applied_drive_command: DriveCommand
    event_bus: List[WarehouseEvent]
    state_start_sim_time: Optional[float]

    @property
    def fsm_status(self) -> FSMStatus: ...

    def assess(
        self,
        frame: Optional[SensorFrame],
        sim_time: float,
        dt: float,
    ) -> RobotTickAssessment: ...

    def commit(self, assessment: RobotTickAssessment) -> None: ...

    def act(self, context: TickContext) -> None: ...

    def assign_task(self, task: Task) -> None: ...

    def set_path(self, path: List[Tuple[int, int]]) -> None: ...

    def release_task_for_requeue(self, task: Task) -> None: ...

    def transition_to(self, state: Any, sim_time: Optional[float] = None) -> None: ...

    def request_fault_reset(
        self,
        fault: FaultCode,
        context: HealthCheckContext,
    ) -> bool: ...

class IRobotRegistry(ABC):
    """Resolve live robots without exposing a mutable agent dictionary."""

    @abstractmethod
    def get(self, robot_id: str) -> Optional[ICoordinatedRobot]:
        raise NotImplementedError

    @abstractmethod
    def values(self) -> Iterable[ICoordinatedRobot]:
        raise NotImplementedError

    @abstractmethod
    def items(self) -> Iterable[Tuple[str, ICoordinatedRobot]]:
        raise NotImplementedError

    @abstractmethod
    def robot_ids(self) -> Tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def register(self, robot_id: str, robot: ICoordinatedRobot) -> None:
        raise NotImplementedError

    @abstractmethod
    def replace(self, robots: Iterable[Tuple[str, ICoordinatedRobot]]) -> None:
        """Replace registry contents; intended for deterministic test fixtures."""
        raise NotImplementedError
