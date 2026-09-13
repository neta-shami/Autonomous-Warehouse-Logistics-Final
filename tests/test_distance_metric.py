"""Physical-distance KPI tests for the metrics-only truth probe."""

import pytest

from entities.pose import Pose
from use_cases.metrics_collector import MetricsCollector


class MutableTruthProbe:
    def __init__(self, positions):
        self.positions = positions

    def true_pose(self, robot_id):
        return self.positions[robot_id]


def test_reports_actual_physical_displacement():
    probe = MutableTruthProbe({"r1": Pose(2.0, 2.0)})
    metrics = MetricsCollector(probe)
    metrics.sample_physical_motion(["r1"], 1.0)
    probe.positions["r1"] = Pose(2.3, 2.0)

    metrics.sample_physical_motion(["r1"], 1.1)

    assert metrics.total_distance_driven == pytest.approx(0.3)


def test_first_truth_sample_is_only_a_baseline():
    metrics = MetricsCollector(MutableTruthProbe({"r1": Pose(2.0, 2.0)}))

    metrics.sample_physical_motion(["r1"], 1.0)

    assert metrics.total_distance_driven == 0.0


def test_stationary_robot_reports_no_movement():
    metrics = MetricsCollector(MutableTruthProbe({"r1": Pose(2.0, 2.0)}))
    metrics.sample_physical_motion(["r1"], 1.0)

    metrics.sample_physical_motion(["r1"], 1.1)

    assert metrics.total_distance_driven == 0.0


def test_diagonal_displacement_uses_euclidean_distance():
    probe = MutableTruthProbe({"r1": Pose(2.0, 2.0)})
    metrics = MetricsCollector(probe)
    metrics.sample_physical_motion(["r1"], 1.0)
    probe.positions["r1"] = Pose(2.3, 2.4)

    metrics.sample_physical_motion(["r1"], 1.1)

    assert metrics.total_distance_driven == pytest.approx(0.5)


def test_external_teleport_resets_baseline_instead_of_counting_motion():
    probe = MutableTruthProbe({"r1": Pose(2.0, 2.0)})
    metrics = MetricsCollector(probe)
    metrics.sample_physical_motion(["r1"], 1.0)
    metrics.mark_external_discontinuity("r1", 1.05)
    probe.positions["r1"] = Pose(8.0, 8.0)

    metrics.sample_physical_motion(["r1"], 1.1)

    assert metrics.total_distance_driven == 0.0
