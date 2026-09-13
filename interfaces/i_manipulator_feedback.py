from abc import ABC, abstractmethod


class IManipulatorFeedback(ABC):
    """Narrow port for checking physical manipulator joint arrival."""

    @abstractmethod
    def joint_at_target(
        self,
        robot_id: str,
        joint_name: str,
        target: float,
        tolerance: float,
    ) -> bool:
        """Return whether a named robot joint is within target tolerance."""
        pass
