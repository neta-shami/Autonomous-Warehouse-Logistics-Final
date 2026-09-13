"""Uncertainty-aware immediate collision braking over a fleet snapshot."""

from math import hypot
from typing import Optional, Tuple

from entities.enums import FSMStatus
from entities.fleet_snapshot import FleetRobotView, FleetSnapshot
from entities.value_validation import require_finite
from interfaces.i_local_collision_avoidance import ILocalCollisionAvoidance


class LocalCollisionAvoidance(ILocalCollisionAvoidance):
    """Brake using bounded footprints, pose error, and relative stopping range."""

    def __init__(
        self,
        physical_margin_m: float = 0.1,
        conservative_deceleration_mps2: float = 1.5,
        worst_case_speed_mps: float = 1.0,
    ):
        for name, value in (
            ("physical_margin_m", physical_margin_m),
            ("conservative_deceleration_mps2", conservative_deceleration_mps2),
            ("worst_case_speed_mps", worst_case_speed_mps),
        ):
            require_finite(name, value, minimum=0.0)
        if conservative_deceleration_mps2 == 0.0:
            raise ValueError("conservative_deceleration_mps2 must be positive")
        self.physical_margin_m = physical_margin_m
        self.conservative_deceleration_mps2 = conservative_deceleration_mps2
        self.worst_case_speed_mps = worst_case_speed_mps

    def check_collisions(
        self, robot_id: str, snapshot: FleetSnapshot
    ) -> Tuple[Optional[FSMStatus], Tuple[str, ...]]:
        robot = snapshot.robot(robot_id)
        if robot.pose_estimate is None:
            return FSMStatus.AVOIDING, ()
        blockers = []
        for other in snapshot.robots:
            if other.robot_id == robot_id:
                continue
            if other.pose_estimate is None:
                return FSMStatus.AVOIDING, ()
            robot_estimate = robot.pose_estimate
            other_estimate = other.pose_estimate
            distance = robot_estimate.pose.distance_to(other_estimate.pose)
            required = (
                robot.footprint_radius_m
                + other.footprint_radius_m
                + robot_estimate.uncertainty_m
                + other_estimate.uncertainty_m
                + self.physical_margin_m
                + self._braking_distance(robot, other)
            )
            if distance < required:
                blockers.append(other.robot_id)
        if blockers:
            return FSMStatus.AVOIDING, tuple(sorted(blockers))
        return None, ()

    def _braking_distance(
        self, robot: FleetRobotView, other: FleetRobotView
    ) -> float:
        if robot.pose_estimate is None or other.pose_estimate is None:
            return self.worst_case_speed_mps**2 / (
                2.0 * self.conservative_deceleration_mps2
            )
        if (
            robot.observed_chassis_velocity is None
            or other.observed_chassis_velocity is None
        ):
            closing_speed = 2.0 * self.worst_case_speed_mps
        else:
            delta_x = other.pose_estimate.pose.x - robot.pose_estimate.pose.x
            delta_y = other.pose_estimate.pose.y - robot.pose_estimate.pose.y
            distance = hypot(delta_x, delta_y)
            if distance == 0.0:
                closing_speed = 0.0
            else:
                relative_x = (
                    robot.observed_chassis_velocity[0]
                    - other.observed_chassis_velocity[0]
                )
                relative_y = (
                    robot.observed_chassis_velocity[1]
                    - other.observed_chassis_velocity[1]
                )
                closing_speed = max(
                    0.0,
                    (relative_x * delta_x + relative_y * delta_y) / distance,
                )
        return closing_speed ** 2 / (
            2.0 * self.conservative_deceleration_mps2
        )
