"""Port for robot-local collision avoidance decisions."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from entities.enums import FSMStatus
from entities.fleet_snapshot import FleetSnapshot


class ILocalCollisionAvoidance(ABC):
    """Return a braking decision from one committed fleet snapshot."""

    @abstractmethod
    def check_collisions(
        self, robot_id: str, snapshot: FleetSnapshot
    ) -> Tuple[Optional[FSMStatus], Tuple[str, ...]]:
        raise NotImplementedError
