from abc import ABC, abstractmethod

from entities.drive_command import DriveCommand
from entities.pose import Pose


class IMotionPredictionModel(ABC):
    """Port for short-term prediction from the command actually applied."""

    @abstractmethod
    def predict(
        self, previous: Pose, command: DriveCommand, dt: float
    ) -> Pose:
        pass
