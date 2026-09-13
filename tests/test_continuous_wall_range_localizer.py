"""Pure tests for continuous absolute position from opposing wall rays."""

from math import inf, nan

import pytest

from entities.enums import CardinalDirection
from entities.sensor_frame import SensorFrame
from entities.warehouse_topology import WarehouseTopology
from use_cases.continuous_wall_range_localizer import (
    ContinuousWallRangeLocalizer,
)


def _topology():
    return WarehouseTopology(10, 10, 0.0, 10.0, 0.0, 10.0, {})


def _frame(wall_ranges, sim_time=2.0):
    return SensorFrame(
        robot_id="r1",
        observed_chassis_velocity=(0.0, 0.0),
        wall_ranges=tuple(wall_ranges),
        payload_range=None,
        sim_time=sim_time,
        sequence=1,
    )


def _ranges(neg_x, pos_x, neg_y, pos_y):
    return (
        (CardinalDirection.NEG_X, neg_x),
        (CardinalDirection.POS_X, pos_x),
        (CardinalDirection.NEG_Y, neg_y),
        (CardinalDirection.POS_Y, pos_y),
    )


def test_locates_an_arbitrary_fractional_pose_without_grid_rounding():
    frame = _frame(_ranges(3.25, 6.75, 7.6, 2.4), sim_time=4.5)

    fix = ContinuousWallRangeLocalizer().locate(frame, _topology())

    assert fix is not None
    assert (fix.pose.x, fix.pose.y) == pytest.approx((3.25, 7.6))
    assert fix.residual_m == pytest.approx(0.0)
    assert fix.sim_time == 4.5


def test_accepts_bounded_opposing_ray_noise_and_reports_residual():
    frame = _frame(_ranges(3.27, 6.76, 7.58, 2.40))
    localizer = ContinuousWallRangeLocalizer(
        pair_residual_tolerance_m=0.05,
        measurement_error_bound_m=0.03,
    )

    fix = localizer.locate(frame, _topology())

    assert fix is not None
    assert (fix.pose.x, fix.pose.y) == pytest.approx((3.255, 7.59))
    assert fix.residual_m == pytest.approx(0.03)
    assert fix.error_bound_m >= 0.03


def test_shortened_occluded_ray_rejects_the_axis_and_whole_fix():
    frame = _frame(_ranges(3.25, 2.0, 7.6, 2.4))

    assert ContinuousWallRangeLocalizer().locate(frame, _topology()) is None


@pytest.mark.parametrize(
    "wall_ranges",
    [
        _ranges(3.25, 6.75, 7.6, 2.4)[:-1],
        _ranges(3.25, 6.75, 7.6, -1.0),
    ],
)
def test_missing_direction_or_no_hit_rejects_the_fix(wall_ranges):
    assert (
        ContinuousWallRangeLocalizer().locate(
            _frame(wall_ranges), _topology()
        )
        is None
    )


@pytest.mark.parametrize("bad_value", [nan, inf, -inf])
def test_non_finite_ray_is_defensively_rejected(bad_value):
    values = {
        CardinalDirection.NEG_X: 3.25,
        CardinalDirection.POS_X: 6.75,
        CardinalDirection.NEG_Y: 7.6,
        CardinalDirection.POS_Y: 2.4,
    }
    values[CardinalDirection.POS_X] = bad_value

    class MalformedFrame:
        sim_time = 1.0

        def range_for(self, direction):
            return values[direction]

    assert (
        ContinuousWallRangeLocalizer().locate(MalformedFrame(), _topology())
        is None
    )


def test_candidate_outside_metric_bounds_is_rejected():
    frame = _frame(_ranges(12.0, 1.0, 5.0, 5.0))
    permissive_residual = ContinuousWallRangeLocalizer(
        pair_residual_tolerance_m=100.0
    )

    assert permissive_residual.locate(frame, _topology()) is None


@pytest.mark.parametrize(
    "bounds",
    [
        (0.0, 0.0, 0.0, 10.0),
        (0.0, 10.0, 5.0, 4.0),
        (0.0, nan, 0.0, 10.0),
    ],
)
def test_topology_rejects_invalid_metric_bounds(bounds):
    with pytest.raises(ValueError):
        WarehouseTopology(10, 10, *bounds, {})
