from abc import ABC, abstractmethod

from entities.sensor_frame import SensorFrame


class ISensorAdapter(ABC):
    """Port for sampling one atomic set of robot-mounted instruments."""

    @abstractmethod
    def read_frame(self, robot_id: str) -> SensorFrame:
        """Read one robot's named sensors without interpreting the values."""
        raise NotImplementedError
