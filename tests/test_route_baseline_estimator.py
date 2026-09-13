from unittest.mock import MagicMock

import pytest

from entities.enums import TaskType
from entities.pose import Pose
from entities.task import Task
from use_cases.route_baseline_estimator import RouteBaselineEstimator


def test_controlled_path_distance_models_waypoint_corner_cutting():
    distance, end = RouteBaselineEstimator.controlled_path_distance(
        Pose(0.0, 0.0),
        [(0, 0), (1, 0), (2, 0)],
        waypoint_tolerance_m=0.5,
        final_tolerance_m=0.1,
    )

    assert distance == pytest.approx(1.9)
    assert end.x == pytest.approx(1.9)
    assert end.y == pytest.approx(0.0)


def test_estimate_combines_source_and_target_paths():
    planner = MagicMock()
    planner.find_path_to_access.side_effect = [
        ((1, 0), [(0, 0), (1, 0)]),
        ((2, 0), [(1, 0), (2, 0)]),
    ]
    topology = MagicMock()
    estimator = RouteBaselineEstimator(planner, topology)
    task = Task.transfer("t1", TaskType.RELOCATE, (1, 0), 0, (2, 0), 0, "p1")

    distance = estimator.estimate(task, Pose(0, 0), 0.35, 0.5, 0.1)

    assert distance == pytest.approx(1.9)
    assert planner.find_path_to_access.call_count == 2
