from abc import ABC, abstractmethod

from entities.pose import Pose


class IGroundTruthProbe(ABC):
    """Metrics-only port for measuring a robot's physical simulator pose."""

    @abstractmethod
    def true_pose(self, robot_id: str) -> Pose:
        """Return physical pose without exposing it to control logic."""
        raise NotImplementedError
