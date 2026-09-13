from types import SimpleNamespace

import pytest

from entities.enums import FSMStatus, TaskType
from entities.events import CollisionStallEvent, TaskCompletedEvent
from entities.pose import Pose
from entities.task import Task
from use_cases.metrics_collector import MetricsCollector


def _task(task_type, task_id="t1"):
    if task_type is TaskType.PARK:
        return Task.park(task_id, (5, 5))
    return Task.transfer(task_id, task_type, (2, 2), 0, (5, 5), 0, "p1")


def test_throughput_counts_a_delivered_package():
    metrics = MetricsCollector()
    metrics.handle_task_completed(
        TaskCompletedEvent(
            sim_time=1.0,
            task_id="t1",
            robot_id="r1",
            success=True,
            task=_task(TaskType.RETRIEVE),
        ))
    assert metrics.total_throughput == 1


def test_throughput_ignores_park_tasks():
    """Driving home is not warehouse throughput.

    Regression test: park tasks publish TaskCompletedEvent like any other task,
    so counting every completion inflated throughput by one per robot per idle
    period.
    """
    metrics = MetricsCollector()
    metrics.handle_task_completed(
        TaskCompletedEvent(
            sim_time=1.0,
            task_id="t1",
            robot_id="r1",
            success=True,
            task=_task(TaskType.PARK),
        ))
    assert metrics.total_throughput == 0


def test_throughput_ignores_failed_tasks():
    """A task that failed did not move a package.

    Regression test: throughput was wired to a handler that incremented
    unconditionally, ignoring the event's success flag.
    """
    metrics = MetricsCollector()
    metrics.handle_task_completed(
        TaskCompletedEvent(
            sim_time=1.0,
            task_id="t1",
            robot_id="r1",
            success=False,
            task=_task(TaskType.RETRIEVE),
        ))
    assert metrics.total_throughput == 0
    assert metrics.total_failed_tasks == 1


def test_throughput_counts_every_transfer_task_type():
    metrics = MetricsCollector()
    for task_type in (TaskType.STORE, TaskType.RETRIEVE, TaskType.RELOCATE):
        task = _task(task_type, task_type.name)
        metrics.handle_task_completed(
            TaskCompletedEvent(
                sim_time=1.0,
                task_id=task.task_id,
                robot_id="r1",
                success=True,
                task=task,
            ))
    assert metrics.total_throughput == 3


def test_duplicate_terminal_event_does_not_inflate_throughput():
    metrics = MetricsCollector()
    task = _task(TaskType.STORE)
    event = TaskCompletedEvent(1.0, task.task_id, "r1", True, task)

    metrics.handle_task_completed(event)
    metrics.handle_task_completed(event)

    assert metrics.total_throughput == 1


def test_terminal_event_deduplication_uses_immutable_attempt_generation():
    metrics = MetricsCollector()
    task = _task(TaskType.STORE)
    task.lease_generation = 1
    event = TaskCompletedEvent(
        1.0,
        task.task_id,
        "r1",
        False,
        task,
        lease_generation=1,
    )

    metrics.handle_task_completed(event)
    task.lease_generation = 2
    metrics.handle_task_completed(event)

    assert metrics.total_failed_tasks == 1


def test_actual_collisions_count_unique_pairs_per_sample():
    metrics = MetricsCollector()

    metrics.record_robot_collisions(
        [("r1", "r2"), ("r1", "r2"), ("r1", "wall_west")]
    )

    assert metrics.total_actual_collisions == 2


def test_report_includes_actual_collision_kpi():
    metrics = MetricsCollector()
    metrics.record_robot_collisions({("r1", "shelf_2_2")})

    assert "Actual Collisions: 1" in metrics.generate_report()


def test_a_proximity_stop_is_recorded_as_a_safety_stop():
    metrics = MetricsCollector()

    metrics.handle_collision_stall(
        CollisionStallEvent(1.0, "r1", (2.0, 2.0), ["r2"])
    )

    assert metrics.near_collision_stops == 1


def test_truth_probe_can_be_injected_only_for_metrics_measurement():
    probe = object()

    metrics = MetricsCollector(truth_probe=probe)

    assert metrics.truth_probe is probe


def test_task_timing_uses_simulation_time_and_transfer_tasks_only():
    metrics = MetricsCollector(nominal_manipulation_time_s=0.2)
    task = _task(TaskType.STORE)
    task.ideal_distance_m = 0.7
    robot = SimpleNamespace(
        state=SimpleNamespace(),
        current_task=task,
        fsm_status=FSMStatus.NAVIGATING,
    )
    metrics.sample_task_progress([robot], sim_time=1.0, dt=0.1)
    robot.fsm_status = FSMStatus.AVOIDING
    metrics.sample_task_progress([robot], sim_time=1.2, dt=0.2)
    robot.fsm_status = FSMStatus.MANIPULATING
    metrics.sample_task_progress([robot], sim_time=1.4, dt=0.2)
    metrics.handle_task_completed(
        TaskCompletedEvent(1.9, task.task_id, "r1", True, task)
    )
    robot.current_task = None
    robot.fsm_status = FSMStatus.IDLE
    metrics.sample_task_progress([robot], sim_time=2.0, dt=0.1)

    assert metrics.average_task_time_s == pytest.approx(1.0)
    assert metrics.average_ideal_task_time_s == pytest.approx(0.9)
    assert metrics.task_time_overhead_ratio == pytest.approx(1 / 9)
    assert metrics.task_success_rate == 1.0


def test_calibrated_ideal_time_does_not_copy_actual_manipulation_delay():
    metrics = MetricsCollector(nominal_manipulation_time_s=0.2)
    task = _task(TaskType.STORE)
    task.ideal_distance_m = 0.7
    robot = SimpleNamespace(
        state=SimpleNamespace(),
        current_task=task,
        fsm_status=FSMStatus.NAVIGATING,
    )
    metrics.sample_task_progress([robot], sim_time=1.0, dt=0.1)
    robot.fsm_status = FSMStatus.MANIPULATING
    metrics.sample_task_progress([robot], sim_time=2.5, dt=1.5)
    metrics.handle_task_completed(
        TaskCompletedEvent(2.9, task.task_id, "r1", True, task)
    )

    assert metrics.average_task_time_s == pytest.approx(2.0)
    assert metrics.average_ideal_task_time_s == pytest.approx(0.9)
    assert metrics.task_time_overhead_ratio == pytest.approx(11 / 9)


def test_transfer_distance_compares_truth_motion_with_ideal_route():
    class Probe:
        pose = Pose(0.0, 0.0)

        def true_pose(self, _robot_id):
            return self.pose

    probe = Probe()
    metrics = MetricsCollector(probe)
    task = _task(TaskType.RETRIEVE)
    task.ideal_distance_m = 0.9
    metrics.sample_physical_motion(["r1"], 0.0, {"r1": task.task_id})
    probe.pose = Pose(1.0, 0.0)
    metrics.sample_physical_motion(["r1"], 1.0, {"r1": task.task_id})
    metrics.handle_task_completed(
        TaskCompletedEvent(1.0, task.task_id, "r1", True, task)
    )

    assert metrics.total_task_distance_driven == pytest.approx(1.0)
    assert metrics.total_ideal_task_distance == pytest.approx(0.9)
    assert metrics.extra_distance_driven_m == pytest.approx(0.1)
    assert metrics.extra_distance_ratio == pytest.approx(1.0 / 9.0)
