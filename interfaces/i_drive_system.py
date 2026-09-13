from abc import ABC, abstractmethod

from entities.drive_command import DriveCommand


class IDriveSystem(ABC):
    @abstractmethod
    def command_velocity(self, robot_id: str, command: DriveCommand) -> None:
        """Apply an already validated and bounded holonomic command."""
        raise NotImplementedError

    @abstractmethod
    def stop(self, robot_id: str) -> None:
        """
        Halts the robot's movement.
        """
        raise NotImplementedError
