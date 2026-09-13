from abc import ABC, abstractmethod


class ILiftSystem(ABC):
    @abstractmethod
    def set_lift_height(self, robot_id: str, height: float) -> None:
        """Controls the vertical position of the arm."""
        pass
