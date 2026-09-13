"""Fault KPI and deterministic injection regression tests."""

from unittest.mock import MagicMock

import pytest

from adapters.fault_injector import FaultInjector
from entities.drive_command import DriveCommand
from entities.enums import (
    CardinalDirection,
    LocalizationStatus,
    PoseSource,
    TaskType,
)
from entities.events import (
    RelocalizationFailedEvent,
    RelocalizedEvent,
)
from entities.inventory_world import InventoryWorld
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from entities.robot_state import RobotState
from entities.sensor_frame import SensorFrame
from entities.task import Task
from entities.warehouse_topology import WarehouseTopology
from interfaces.sensor_read_error import SensorReadError
from use_cases.continuous_wall_range_localizer import ContinuousWallRangeLocalizer
from use_cases.metrics_collector import MetricsCollector


class MutableTruthProbe:
    def __init__(self, pose):
        self.pose = pose

    def true_pose(self, _robot_id):
        return self.pose


class FrameSource:
    def __init__(self, frame):
        self.frame = frame

    def read_frame(self, _robot_id):
        return self.frame


def _frame(sequence=0):
    return SensorFrame(
        "r1",
        (0.2, 0.0),
        (
            (CardinalDirection.POS_X, 8.0),
            (CardinalDirection.NEG_X, 2.0),
            (CardinalDirection.POS_Y, 7.0),
            (CardinalDirection.NEG_Y, 3.0),
        ),
        -1.0,
        1.0 + sequence * 0.02,
        sequence,
    )


def _state(pose, source=PoseSource.FUSED):
    return RobotState(
        "r1",
        PoseEstimate(pose, 0.03, source, 1.0),
        (0.0, 0.0),
        None,
        False,
        False,
        (1, 9),
        LocalizationStatus.TRUSTED,
    )


def test_external_recovery_metrics_capture_paired_outcome_and_duration():
    metrics = MetricsCollector()
    new = Pose(5.0, 5.0)

    metrics.mark_external_discontinuity("r1", 1.0)
    metrics.handle_relocalized(RelocalizedEvent(1.08, "r1", new, 0.06))

    assert metrics.false_relocations == 0
    assert metrics.automatic_recovery_rate == 1.0
    assert metrics.mean_recovery_time_s == pytest.approx(0.06)


def test_recovery_rate_pairs_each_recovery_with_one_injected_trial():
    metrics = MetricsCollector()
    metrics.mark_external_discontinuity("r1", 1.0)
    metrics.mark_external_discontinuity("r1", 2.0)

    metrics.handle_relocalized(
        RelocalizedEvent(2.1, "r1", Pose(2.0, 2.0), 0.1)
    )
    metrics.handle_relocalization_failed(
        RelocalizationFailedEvent(2.2, "r1")
    )
    metrics.handle_relocalized(
        RelocalizedEvent(2.3, "r1", Pose(2.0, 2.0), 0.1)
    )

    assert metrics.automatic_recovery_rate == pytest.approx(0.5)
    assert metrics.recovered_external_discontinuities == 1
    assert metrics.false_relocations == 1


def test_localization_metrics_report_error_grid_accuracy_and_fix_availability():
    truth = MutableTruthProbe(Pose(2.02, 2.01))
    metrics = MetricsCollector(truth)

    metrics.sample_localization_error(
        [_state(Pose(2.0, 2.0), PoseSource.FUSED)], sim_time=0.1
    )
    truth.pose = Pose(2.49, 2.49)
    metrics.sample_localization_error(
        [_state(Pose(2.51, 2.51), PoseSource.PREDICTED)], sim_time=0.2
    )

    assert metrics.mean_localization_error_m < 0.04
    assert metrics.peak_localization_error_m < 0.04
    assert metrics.grid_cell_accuracy == pytest.approx(0.5)
    assert metrics.absolute_fix_availability == pytest.approx(0.5)
    assert metrics.initialization_success_rate == 1.0


def test_command_suppression_records_zero_additional_unsafe_batches():
    metrics = MetricsCollector()

    metrics.record_command_result(
        1.0, unsafe=True, command=DriveCommand(0.0, 0.0)
    )
    metrics.record_command_result(
        1.02, unsafe=True, command=DriveCommand(0.1, 0.0)
    )

    assert metrics.unsafe_command_batches == 1


def test_orphan_lease_metric_distinguishes_owned_recovery_lease():
    world = InventoryWorld()
    task = Task.transfer(
        "t1", TaskType.STORE, (1, 1), 0, (2, 2), 0, "p1"
    )
    assert world.acquire_task_lease(task) is not None
    metrics = MetricsCollector()

    metrics.sample_orphan_leases(world, active_task_ids={"t1"})
    assert metrics.orphan_lease_count == 0
    metrics.sample_orphan_leases(world, active_task_ids=set())
    assert metrics.orphan_lease_count == 1


def test_fault_injector_marks_discontinuity_immediately_before_teleport():
    order = []
    metrics = MagicMock()
    metrics.mark_external_discontinuity.side_effect = (
        lambda *_args: order.append("marked")
    )
    injector = FaultInjector(FrameSource(_frame()), metrics, random_seed=7)

    injector.teleport_robot(
        "r1", Pose(8.0, 8.0), 1.0, lambda *_args: order.append("teleported")
    )

    assert order == ["marked", "teleported"]


def test_fault_injector_dropout_occlusion_and_seeded_bias_are_repeatable():
    first = FaultInjector(FrameSource(_frame()), MetricsCollector(), random_seed=4)
    second = FaultInjector(FrameSource(_frame()), MetricsCollector(), random_seed=4)
    first.drop_next_frames("r1", 1)
    with pytest.raises(SensorReadError):
        first.read_frame("r1")

    first.occlude_ray("r1", CardinalDirection.POS_X)
    assert first.read_frame("r1").range_for(CardinalDirection.POS_X) is None
    first.clear_sensor_faults("r1")
    first.inject_seeded_single_ray_bias("r1", 0.2)
    second.inject_seeded_single_ray_bias("r1", 0.2)

    assert first.read_frame("r1").wall_ranges == second.read_frame("r1").wall_ranges


def test_fault_injector_applies_a_fixed_validated_range_override():
    injector = FaultInjector(FrameSource(_frame()), MetricsCollector(), random_seed=3)

    injector.override_ray("r1", CardinalDirection.POS_X, 4.25)

    assert injector.read_frame("r1").range_for(
        CardinalDirection.POS_X
    ) == pytest.approx(4.25)


def test_gradual_single_ray_bias_breaks_opposing_pair_residual():
    injector = FaultInjector(FrameSource(_frame()), MetricsCollector(), random_seed=1)
    injector.set_gradual_ray_bias(
        "r1", CardinalDirection.POS_X, step_m=0.06
    )

    first = injector.read_frame("r1")
    second = injector.read_frame("r1")

    assert first.range_for(CardinalDirection.POS_X) == pytest.approx(8.06)
    assert second.range_for(CardinalDirection.POS_X) == pytest.approx(8.12)
    localizer = ContinuousWallRangeLocalizer(
        pair_residual_tolerance_m=0.1,
        measurement_error_bound_m=0.01,
    )
    topology = WarehouseTopology(10, 10, 0.0, 10.0, 0.0, 10.0, {})
    assert localizer.locate(first, topology) is not None
    assert localizer.locate(second, topology) is None
