from abc import ABC, abstractmethod


class IExtendSystem(ABC):
    @abstractmethod
    def extend_arm(self, robot_id: str, extension: float) -> None:
        """Controls the horizontal extension of the arm into the shelf."""
        pass
