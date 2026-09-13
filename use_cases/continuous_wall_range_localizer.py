from math import isfinite, sqrt
from typing import Optional, Tuple

from entities.absolute_pose_fix import AbsolutePoseFix
from entities.enums import CardinalDirection
from entities.pose import Pose
from entities.sensor_frame import SensorFrame
from entities.value_validation import require_finite
from entities.warehouse_topology import WarehouseTopology
from interfaces.i_absolute_localizer import IAbsoluteLocalizer


class ContinuousWallRangeLocalizer(IAbsoluteLocalizer):
    """Solve a continuous pose from two independent opposing wall pairs."""

    def __init__(
        self,
        pair_residual_tolerance_m: float = 0.1,
        measurement_error_bound_m: float = 0.05,
    ):
        require_finite(
            "pair_residual_tolerance_m",
            pair_residual_tolerance_m,
            minimum=0.0,
        )
        require_finite(
            "measurement_error_bound_m",
            measurement_error_bound_m,
            minimum=0.0,
        )
        self.pair_residual_tolerance_m = pair_residual_tolerance_m
        self.measurement_error_bound_m = measurement_error_bound_m

    def locate(
        self, frame: SensorFrame, topology: WarehouseTopology
    ) -> Optional[AbsolutePoseFix]:
        x_axis = self._locate_axis(
            frame,
            CardinalDirection.NEG_X,
            CardinalDirection.POS_X,
            topology.min_x,
            topology.max_x,
        )
        y_axis = self._locate_axis(
            frame,
            CardinalDirection.NEG_Y,
            CardinalDirection.POS_Y,
            topology.min_y,
            topology.max_y,
        )
        if x_axis is None or y_axis is None:
            return None

        try:
            frame_time_is_valid = (
                not isinstance(frame.sim_time, bool)
                and isfinite(frame.sim_time)
                and frame.sim_time >= 0.0
            )
        except (AttributeError, TypeError):
            return None
        if not frame_time_is_valid:
            return None

        x, x_residual = x_axis
        y, y_residual = y_axis
        residual = max(x_residual, y_residual)
        per_axis_bound = self.measurement_error_bound_m + residual / 2.0
        return AbsolutePoseFix(
            pose=Pose(x, y),
            error_bound_m=sqrt(2.0) * per_axis_bound,
            residual_m=residual,
            sim_time=frame.sim_time,
        )

    def _locate_axis(
        self,
        frame: SensorFrame,
        negative_direction: CardinalDirection,
        positive_direction: CardinalDirection,
        minimum: float,
        maximum: float,
    ) -> Optional[Tuple[float, float]]:
        try:
            negative_range = frame.range_for(negative_direction)
            positive_range = frame.range_for(positive_direction)
        except (AttributeError, KeyError, TypeError):
            return None
        if not self._is_positive_finite(negative_range) or not self._is_positive_finite(
            positive_range
        ):
            return None
        assert negative_range is not None and positive_range is not None

        span = maximum - minimum
        residual = abs((negative_range + positive_range) - span)
        if residual > self.pair_residual_tolerance_m:
            return None

        from_negative = minimum + negative_range
        from_positive = maximum - positive_range
        coordinate = (from_negative + from_positive) / 2.0
        if not minimum <= coordinate <= maximum:
            return None
        return coordinate, residual

    @staticmethod
    def _is_positive_finite(value: Optional[float]) -> bool:
        if value is None or isinstance(value, bool):
            return False
        try:
            return isfinite(value) and value > 0.0
        except TypeError:
            return False
