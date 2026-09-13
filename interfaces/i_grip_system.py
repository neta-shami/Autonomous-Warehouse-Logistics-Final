from abc import ABC, abstractmethod


class IGripSystem(ABC):
    @abstractmethod
    def grip(
        self, robot_id: str, expected_payload_id: str, engage: bool
    ) -> None:
        """Engages or disengages the gripper to hold a package."""
        pass
