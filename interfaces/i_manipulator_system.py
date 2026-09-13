from abc import ABC, abstractmethod

from entities.enums import (
    CardinalDirection,
    ManipulatorAbortResult,
    ManipulatorResult,
)


class IManipulatorSystem(ABC):
    @abstractmethod
    def pick(
        self, robot_id: str, expected_payload_id: str, tier: int,
        approach: CardinalDirection
    ) -> ManipulatorResult:
        """
        Pick from a tier while facing the target's approach direction.
        Return typed progress, success, or failure.
        """
        pass

    @abstractmethod
    def drop(
        self, robot_id: str, expected_payload_id: str, tier: int,
        approach: CardinalDirection
    ) -> ManipulatorResult:
        """
        Drop onto a tier while facing the target's approach direction.
        Return typed progress, success, or failure.
        """
        pass

    @abstractmethod
    def abort(self, robot_id: str) -> ManipulatorAbortResult:
        """Stop an operation and report whether physical reconciliation is needed."""
        pass
