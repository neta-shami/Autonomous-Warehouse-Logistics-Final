"""Static route-distance baseline used only for KPI evaluation."""

from math import hypot
from typing import List, Tuple

from entities.pose import Pose
from entities.task import Task
from entities.warehouse_topology import WarehouseTopology
from interfaces.i_path_planner import IPathPlanner
from interfaces.i_route_baseline_estimator import IRouteBaselineEstimator


class RouteBaselineEstimator(IRouteBaselineEstimator):
    """Measure the no-traffic path produced by the normal waypoint policy."""

    def __init__(
        self,
        path_planner: IPathPlanner,
        topology: WarehouseTopology,
    ) -> None:
        self.path_planner = path_planner
        self.topology = topology

    def estimate(
        self,
        task: Task,
        start_pose: Pose,
        footprint_radius_m: float,
        waypoint_tolerance_m: float,
        final_tolerance_m: float,
    ) -> float:
        """Calculate source and target legs without traffic reservations."""
        if task.source_coords is None:
            raise ValueError("route baseline requires a transfer task source")
        source_access, source_path = self.path_planner.find_path_to_access(
            start_pose,
            task.source_coords,
            self.topology,
            (),
            footprint_radius_m=footprint_radius_m,
        )
        if source_access is None or not source_path:
            return 0.0
        source_distance, target_start = self.controlled_path_distance(
            start_pose,
            source_path,
            waypoint_tolerance_m,
            final_tolerance_m,
        )
        target_access, target_path = self.path_planner.find_path_to_access(
            target_start,
            task.target_coords,
            self.topology,
            (),
            footprint_radius_m=footprint_radius_m,
        )
        if target_access is None or not target_path:
            return 0.0
        target_distance, _ = self.controlled_path_distance(
            target_start,
            target_path,
            waypoint_tolerance_m,
            final_tolerance_m,
        )
        return source_distance + target_distance

    @staticmethod
    def controlled_path_distance(
        start: Pose,
        path: List[Tuple[int, int]],
        waypoint_tolerance_m: float,
        final_tolerance_m: float,
    ) -> Tuple[float, Pose]:
        """Integrate the route length after normal waypoint corner cutting."""
        distance = 0.0
        current_x, current_y = start.x, start.y
        final_index = len(path) - 1
        for index, waypoint in enumerate(path):
            dx = waypoint[0] - current_x
            dy = waypoint[1] - current_y
            separation = hypot(dx, dy)
            tolerance = (
                final_tolerance_m if index == final_index else waypoint_tolerance_m
            )
            leg_distance = max(0.0, separation - tolerance)
            if separation > 0.0 and leg_distance > 0.0:
                current_x += dx * leg_distance / separation
                current_y += dy * leg_distance / separation
                distance += leg_distance
        return distance, Pose(current_x, current_y)
