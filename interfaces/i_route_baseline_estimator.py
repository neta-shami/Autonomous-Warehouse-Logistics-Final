"""Port for calculating a task's fault-free route-distance baseline."""

from abc import ABC, abstractmethod

from entities.pose import Pose
from entities.task import Task


class IRouteBaselineEstimator(ABC):
    """Calculate evaluation baselines without coupling fleet policy to geometry."""

    @abstractmethod
    def estimate(
        self,
        task: Task,
        start_pose: Pose,
        footprint_radius_m: float,
        waypoint_tolerance_m: float,
        final_tolerance_m: float,
    ) -> float:
        """Return the ideal source-plus-target distance, or zero if unavailable."""
        pass
