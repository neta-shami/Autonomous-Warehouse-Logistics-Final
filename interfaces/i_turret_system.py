from abc import ABC, abstractmethod

from entities.enums import CardinalDirection


class ITurretSystem(ABC):
    """Port for aiming an arm in a world-aligned direction."""

    @abstractmethod
    def aim_turret(
        self, robot_id: str, approach: CardinalDirection
    ) -> None:
        """Point the arm from its access cell toward the payload."""
        pass
