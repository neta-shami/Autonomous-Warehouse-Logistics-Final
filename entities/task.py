from dataclasses import dataclass
from typing import Optional, Tuple

from .enums import TaskExecutionPhase, TaskStatus, TaskType
from .value_validation import require_finite, require_nonnegative_integer


@dataclass
class Task:
    task_id: str
    task_type: TaskType
    source_coords: Optional[Tuple[int, int]]
    source_tier: Optional[int]
    target_coords: Tuple[int, int]
    target_tier: int
    status: TaskStatus = TaskStatus.PENDING
    assigned_robot_id: Optional[str] = None
    payload_id: Optional[str] = None
    lease_generation: int = 0
    phase: Optional[TaskExecutionPhase] = None
    ideal_distance_m: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.task_type, TaskType):
            raise TypeError("task_type must be TaskType")
        self._validate_coords("target_coords", self.target_coords)
        require_nonnegative_integer("target_tier", self.target_tier)
        require_nonnegative_integer("lease_generation", self.lease_generation)
        if self.ideal_distance_m is not None:
            require_finite(
                "ideal_distance_m", self.ideal_distance_m, minimum=0.0
            )
        if self.payload_id is not None and (
            not isinstance(self.payload_id, str) or not self.payload_id
        ):
            raise ValueError("payload_id must be a non-empty string or None")
        if self.task_type is TaskType.PARK:
            if self.payload_id is not None:
                raise ValueError("PARK tasks cannot have a payload")
            if self.source_coords is not None or self.source_tier is not None:
                raise ValueError("PARK tasks cannot have a source leg")
            if self.phase is None:
                self.phase = TaskExecutionPhase.TO_TARGET
        else:
            self._validate_coords("source_coords", self.source_coords)
            if self.source_tier is None:
                raise TypeError("source_tier must be an integer for transfer tasks")
            require_nonnegative_integer("source_tier", self.source_tier)
            if self.phase is None:
                self.phase = TaskExecutionPhase.TO_SOURCE
        if not isinstance(self.phase, TaskExecutionPhase):
            raise TypeError("phase must be TaskExecutionPhase")
        if not isinstance(self.status, TaskStatus):
            raise TypeError("status must be TaskStatus")
        if self.assigned_robot_id is not None and (
            not isinstance(self.assigned_robot_id, str)
            or not self.assigned_robot_id
        ):
            raise ValueError(
                "assigned_robot_id must be a non-empty string or None"
            )

    @classmethod
    def transfer(
        cls,
        task_id: str,
        task_type: TaskType,
        source_coords: Tuple[int, int],
        source_tier: int,
        target_coords: Tuple[int, int],
        target_tier: int,
        payload_id: Optional[str] = None,
    ) -> "Task":
        if task_type not in (
            TaskType.STORE,
            TaskType.RETRIEVE,
            TaskType.RELOCATE,
        ):
            raise ValueError("transfer factory requires a transfer task type")
        return cls(
            task_id,
            task_type,
            source_coords,
            source_tier,
            target_coords,
            target_tier,
            payload_id=payload_id,
            phase=TaskExecutionPhase.TO_SOURCE,
        )

    @classmethod
    def park(cls, task_id: str, target_coords: Tuple[int, int]) -> "Task":
        return cls(
            task_id,
            TaskType.PARK,
            None,
            None,
            target_coords,
            0,
            phase=TaskExecutionPhase.TO_TARGET,
        )

    @staticmethod
    def _validate_coords(name: str, coords: Optional[Tuple[int, int]]) -> None:
        if (
            not isinstance(coords, tuple)
            or len(coords) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in coords)
        ):
            raise TypeError(f"{name} must be an integer x/y tuple")

    def update_status(self, new_status: TaskStatus) -> None:
        if not isinstance(new_status, TaskStatus):
            raise TypeError("new_status must be TaskStatus")
        self.status = new_status
