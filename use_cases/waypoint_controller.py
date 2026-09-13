"""Belief-based holonomic waypoint control."""

import math
from typing import Tuple

from entities.drive_command import DriveCommand
from entities.pose import Pose
from entities.value_validation import require_finite


class WaypointController:
    """Convert a believed pose and waypoint into one bounded drive command."""

    def __init__(
        self, max_speed_mps: float, proportional_gain: float = 2.0
    ):
        require_finite("max_speed_mps", max_speed_mps)
        require_finite("proportional_gain", proportional_gain)
        if max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be greater than zero")
        if proportional_gain <= 0.0:
            raise ValueError("proportional_gain must be greater than zero")
        self.max_speed_mps = max_speed_mps
        self.proportional_gain = proportional_gain

    def command_for(
        self,
        believed_pose: Pose,
        waypoint: Tuple[float, float],
        arrival_tolerance_m: float,
    ) -> DriveCommand:
        """Return zero on arrival, otherwise a vector-speed-limited command."""
        if not isinstance(believed_pose, Pose):
            raise TypeError("believed_pose must be a Pose")
        if not isinstance(waypoint, tuple) or len(waypoint) != 2:
            raise TypeError("waypoint must be an immutable x/y tuple")
        require_finite("waypoint x", waypoint[0])
        require_finite("waypoint y", waypoint[1])
        require_finite(
            "arrival_tolerance_m", arrival_tolerance_m, minimum=0.0
        )

        delta_x = waypoint[0] - believed_pose.x
        delta_y = waypoint[1] - believed_pose.y
        distance = math.hypot(delta_x, delta_y)
        if distance <= arrival_tolerance_m:
            return DriveCommand(0.0, 0.0)

        raw_vx = self.proportional_gain * delta_x
        raw_vy = self.proportional_gain * delta_y
        raw_speed = math.hypot(raw_vx, raw_vy)
        scale = min(1.0, self.max_speed_mps / raw_speed)
        return DriveCommand(raw_vx * scale, raw_vy * scale)
