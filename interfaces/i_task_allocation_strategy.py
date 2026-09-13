from abc import ABC, abstractmethod
from typing import Dict, List

from entities.robot_state import RobotState
from entities.task import Task


class ITaskAllocationStrategy(ABC):
    @abstractmethod
    def allocate_tasks(self, unassigned_tasks: List[Task], idle_robots: List[RobotState]) -> Dict[str, str]:
        """
        Matches tasks to robots.
        Returns a dictionary mapping task_id to robot_id.
        """
        pass
