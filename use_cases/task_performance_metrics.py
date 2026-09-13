"""Task completion, timing, and route-efficiency measurements."""

from typing import Dict, Iterable, Optional, Protocol, Set, Tuple

from entities.enums import TaskType
from entities.events import TaskCompletedEvent
from entities.task import Task
from entities.value_validation import require_finite


class ActiveTaskRobot(Protocol):
    """Narrow view required to observe active transfer work."""

    current_task: Optional[Task]


class TaskPerformanceMetrics:
    """Own every task-level KPI and its per-attempt bookkeeping."""

    def __init__(
        self,
        nominal_drive_speed_mps: float,
        nominal_manipulation_time_s: float,
    ) -> None:
        require_finite(
            "nominal_drive_speed_mps",
            nominal_drive_speed_mps,
            minimum=1e-12,
        )
        require_finite(
            "nominal_manipulation_time_s",
            nominal_manipulation_time_s,
            minimum=0.0,
        )
        self.nominal_drive_speed_mps = nominal_drive_speed_mps
        self.nominal_manipulation_time_s = nominal_manipulation_time_s
        self.total_throughput = 0
        self.total_failed_tasks = 0
        self.successful_task_durations_s: list[float] = []
        self.successful_ideal_task_durations_s: list[float] = []
        self.total_task_distance_driven = 0.0
        self.total_ideal_task_distance = 0.0
        self._task_started_at: Dict[str, float] = {}
        self._actual_distance_by_task: Dict[str, float] = {}
        self._terminal_task_attempts: Set[Tuple[str, int]] = set()
        self.successful_robot_ids: Set[str] = set()

    def sample_progress(
        self,
        robots: Iterable[ActiveTaskRobot],
        sim_time: float,
        dt: float,
    ) -> None:
        """Record the first active interval of each transfer task."""
        require_finite("task sample sim_time", sim_time, minimum=0.0)
        require_finite("task sample dt", dt, minimum=0.0)
        for robot in robots:
            task = robot.current_task
            if task is not None and task.task_type is not TaskType.PARK:
                self._task_started_at.setdefault(task.task_id, sim_time - dt)

    def record_displacement(self, task_id: str, displacement_m: float) -> None:
        """Attribute one measured physical displacement to its active task."""
        require_finite("task displacement", displacement_m, minimum=0.0)
        self._actual_distance_by_task[task_id] = (
            self._actual_distance_by_task.get(task_id, 0.0) + displacement_m
        )

    def handle_task_completed(self, event: TaskCompletedEvent) -> None:
        """Record one terminal transfer attempt idempotently."""
        attempt = (event.task_id, event.lease_generation)
        if attempt in self._terminal_task_attempts:
            return
        self._terminal_task_attempts.add(attempt)
        task = event.task
        task_type = None if task is None else task.task_type
        started_at = self._task_started_at.pop(event.task_id, None)
        actual_distance = self._actual_distance_by_task.pop(event.task_id, 0.0)
        if event.success and task_type is not TaskType.PARK and started_at is not None:
            self._record_success_duration(event, started_at)
        if task_type is TaskType.PARK:
            return
        if not event.success:
            self.total_failed_tasks += 1
            return
        self.successful_robot_ids.add(event.robot_id)
        self.total_task_distance_driven += actual_distance
        ideal_distance = None if task is None else task.ideal_distance_m
        if ideal_distance is not None:
            self.total_ideal_task_distance += ideal_distance
        self.total_throughput += 1

    def _record_success_duration(
        self,
        event: TaskCompletedEvent,
        started_at: float,
    ) -> None:
        self.successful_task_durations_s.append(max(0.0, event.sim_time - started_at))
        task = event.task
        ideal_distance = None if task is None else task.ideal_distance_m
        if ideal_distance is not None:
            self.successful_ideal_task_durations_s.append(
                ideal_distance / self.nominal_drive_speed_mps
                + self.nominal_manipulation_time_s
            )

    @property
    def task_success_rate(self) -> float:
        attempts = self.total_throughput + self.total_failed_tasks
        return 0.0 if attempts == 0 else self.total_throughput / attempts

    @property
    def average_task_time_s(self) -> float:
        if not self.successful_task_durations_s:
            return 0.0
        return sum(self.successful_task_durations_s) / len(
            self.successful_task_durations_s
        )

    @property
    def average_ideal_task_time_s(self) -> float:
        if not self.successful_ideal_task_durations_s:
            return 0.0
        return sum(self.successful_ideal_task_durations_s) / len(
            self.successful_ideal_task_durations_s
        )

    @property
    def task_time_overhead_ratio(self) -> float:
        ideal = sum(self.successful_ideal_task_durations_s)
        if ideal == 0.0:
            return 0.0
        return max(0.0, sum(self.successful_task_durations_s) - ideal) / ideal

    @property
    def extra_distance_driven_m(self) -> float:
        return max(
            0.0,
            self.total_task_distance_driven - self.total_ideal_task_distance,
        )

    @property
    def extra_distance_ratio(self) -> float:
        if self.total_ideal_task_distance == 0.0:
            return 0.0
        return self.extra_distance_driven_m / self.total_ideal_task_distance
