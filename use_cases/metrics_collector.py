"""Facade over focused warehouse metric recorders."""

from typing import Dict, Iterable, Optional, Tuple

from entities.drive_command import DriveCommand
from entities.events import (
    CollisionStallEvent,
    RelocalizationFailedEvent,
    RelocalizedEvent,
    TaskCompletedEvent,
)
from entities.inventory_world import InventoryWorld
from entities.robot_state import RobotState
from interfaces.i_ground_truth_probe import IGroundTruthProbe
from use_cases.ground_truth_metrics import GroundTruthMetrics
from use_cases.recovery_metrics import RecoveryMetrics
from use_cases.safety_metrics import SafetyMetrics
from use_cases.task_performance_metrics import (
    ActiveTaskRobot,
    TaskPerformanceMetrics,
)


class MetricsCollector:
    """Route samples and events to their single-purpose metric owner."""

    def __init__(
        self,
        truth_probe: Optional[IGroundTruthProbe] = None,
        *,
        nominal_drive_speed_mps: float = 1.0,
        nominal_manipulation_time_s: float = 1.5,
    ) -> None:
        self._task = TaskPerformanceMetrics(
            nominal_drive_speed_mps,
            nominal_manipulation_time_s,
        )
        self._ground_truth = GroundTruthMetrics(truth_probe)
        self._safety = SafetyMetrics()
        self._recovery = RecoveryMetrics()

    def mark_external_discontinuity(self, robot_id: str, sim_time: float) -> None:
        """Reset motion sampling and start one paired recovery trial."""
        self._recovery.mark_external_discontinuity(robot_id, sim_time)
        self._ground_truth.reset_motion_baseline(robot_id)

    def sample_physical_motion(
        self,
        robot_ids: Iterable[str],
        sim_time: float,
        active_task_ids: Optional[Dict[str, str]] = None,
    ) -> None:
        """Measure movement and attribute active displacement to task KPIs."""
        displacements = self._ground_truth.sample_motion(robot_ids, sim_time)
        task_ids = active_task_ids or {}
        for robot_id, displacement in displacements.items():
            task_id = task_ids.get(robot_id)
            if task_id is not None:
                self._task.record_displacement(task_id, displacement)

    def sample_localization_error(
        self,
        robots: Iterable[RobotState],
        sim_time: Optional[float] = None,
    ) -> None:
        self._ground_truth.sample_localization(
            robots,
            self._recovery.pending_robot_ids,
            sim_time,
        )

    def sample_task_progress(
        self,
        robots: Iterable[ActiveTaskRobot],
        sim_time: float,
        dt: float,
    ) -> None:
        self._task.sample_progress(robots, sim_time, dt)

    def record_command_result(
        self,
        sim_time: float,
        *,
        unsafe: bool,
        command: DriveCommand,
    ) -> None:
        self._safety.record_command_result(
            sim_time,
            unsafe=unsafe,
            command=command,
        )

    def sample_orphan_leases(
        self,
        inventory_world: InventoryWorld,
        active_task_ids: Iterable[str],
    ) -> None:
        self._safety.sample_orphan_leases(inventory_world, active_task_ids)

    def record_robot_collisions(
        self, collision_pairs: Iterable[Tuple[str, str]]
    ) -> None:
        self._safety.record_robot_collisions(collision_pairs)

    def handle_collision_stall(self, event: CollisionStallEvent) -> None:
        self._safety.record_near_collision_stop()

    def handle_relocalized(self, event: RelocalizedEvent) -> None:
        self._recovery.handle_relocalized(event)

    def handle_relocalization_failed(self, event: RelocalizationFailedEvent) -> None:
        self._recovery.handle_relocalization_failed(event)

    def handle_task_completed(self, event: TaskCompletedEvent) -> None:
        self._task.handle_task_completed(event)

    @property
    def truth_probe(self) -> Optional[IGroundTruthProbe]:
        return self._ground_truth.truth_probe

    @property
    def total_throughput(self) -> int:
        return self._task.total_throughput

    @property
    def total_failed_tasks(self) -> int:
        return self._task.total_failed_tasks

    @property
    def successful_robot_ids(self) -> frozenset[str]:
        """Robots that completed at least one non-parking transfer."""
        return frozenset(self._task.successful_robot_ids)

    @property
    def successful_task_durations_s(self) -> Tuple[float, ...]:
        return tuple(self._task.successful_task_durations_s)

    @property
    def successful_ideal_task_durations_s(self) -> Tuple[float, ...]:
        return tuple(self._task.successful_ideal_task_durations_s)

    @property
    def task_success_rate(self) -> float:
        return self._task.task_success_rate

    @property
    def average_task_time_s(self) -> float:
        return self._task.average_task_time_s

    @property
    def average_ideal_task_time_s(self) -> float:
        return self._task.average_ideal_task_time_s

    @property
    def task_time_overhead_ratio(self) -> float:
        return self._task.task_time_overhead_ratio

    @property
    def total_task_distance_driven(self) -> float:
        return self._task.total_task_distance_driven

    @property
    def total_ideal_task_distance(self) -> float:
        return self._task.total_ideal_task_distance

    @property
    def extra_distance_driven_m(self) -> float:
        return self._task.extra_distance_driven_m

    @property
    def extra_distance_ratio(self) -> float:
        return self._task.extra_distance_ratio

    @property
    def total_distance_driven(self) -> float:
        return self._ground_truth.total_distance_driven

    @property
    def localization_errors(self) -> Tuple[float, ...]:
        return tuple(self._ground_truth.localization_errors)

    @property
    def mean_localization_error_m(self) -> float:
        return self._ground_truth.mean_localization_error_m

    @property
    def peak_localization_error_m(self) -> float:
        return self._ground_truth.peak_localization_error_m

    @property
    def grid_cell_accuracy(self) -> float:
        return self._ground_truth.grid_cell_accuracy

    @property
    def absolute_fix_availability(self) -> float:
        return self._ground_truth.absolute_fix_availability

    @property
    def initialization_success_rate(self) -> float:
        return self._ground_truth.initialization_success_rate

    @property
    def initialization_durations_s(self) -> Dict[str, float]:
        return dict(self._ground_truth.initialization_durations_s)

    @property
    def near_collision_stops(self) -> int:
        return self._safety.near_collision_stops

    @property
    def total_actual_collisions(self) -> int:
        return self._safety.total_actual_collisions

    @property
    def unsafe_command_batches(self) -> int:
        return self._safety.unsafe_command_batches

    @property
    def orphan_lease_count(self) -> int:
        return self._safety.orphan_lease_count

    @property
    def peak_orphan_lease_count(self) -> int:
        return self._safety.peak_orphan_lease_count

    @property
    def total_external_discontinuities(self) -> int:
        return self._recovery.total_external_discontinuities

    @property
    def recovered_external_discontinuities(self) -> int:
        return self._recovery.recovered_external_discontinuities

    @property
    def false_relocations(self) -> int:
        return self._recovery.false_relocations

    @property
    def automatic_recovery_rate(self) -> float:
        return self._recovery.automatic_recovery_rate

    @property
    def mean_recovery_time_s(self) -> float:
        return self._recovery.mean_recovery_time_s

    def generate_report(self) -> str:
        lines = [
            "Core KPI Report:",
            (
                f"- Completed / Failed Tasks: {self.total_throughput} / "
                f"{self.total_failed_tasks}"
            ),
            f"- Task Success Rate: {self.task_success_rate:.2%}",
            (
                "- Task Time (Actual / Calibrated Ideal / Overhead): "
                f"{self.average_task_time_s:.3f} / "
                f"{self.average_ideal_task_time_s:.3f} s "
                f"({self.task_time_overhead_ratio:.2%} overhead)"
            ),
            (
                "- Transfer Distance (Actual / Ideal / Extra): "
                f"{self.total_task_distance_driven:.2f} / "
                f"{self.total_ideal_task_distance:.2f} / "
                f"{self.extra_distance_driven_m:.2f} m "
                f"({self.extra_distance_ratio:.2%} extra)"
            ),
            (
                f"- Traffic Proximity Stops: {self.near_collision_stops} / "
                f"Actual Collisions: {self.total_actual_collisions}"
            ),
            f"- Mean Localization Error: {self.mean_localization_error_m:.4f} m",
            f"- Unsafe Command Batches: {self.unsafe_command_batches}",
            (
                "- Orphan Leases (Current / Peak): "
                f"{self.orphan_lease_count} / {self.peak_orphan_lease_count}"
            ),
        ]
        if self.total_external_discontinuities:
            lines.append(
                "- Automatic Recovery Rate / Mean Time: "
                f"{self.automatic_recovery_rate:.2%} / "
                f"{self.mean_recovery_time_s:.3f} s"
            )
        return "\n".join(lines)
