from abc import ABC, abstractmethod
from typing import Optional

from entities.absolute_pose_fix import AbsolutePoseFix
from entities.sensor_frame import SensorFrame
from entities.warehouse_topology import WarehouseTopology


class IAbsoluteLocalizer(ABC):
    """Port for stateless absolute localization from one sensor frame."""

    @abstractmethod
    def locate(
        self, frame: SensorFrame, topology: WarehouseTopology
    ) -> Optional[AbsolutePoseFix]:
        """Return a trustworthy continuous 2D fix, or None."""
        pass
