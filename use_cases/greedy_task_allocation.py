from typing import Dict, List

from entities.robot_state import RobotState
from entities.task import Task
from interfaces.i_task_allocation_strategy import ITaskAllocationStrategy


class GreedyTaskAllocationStrategy(ITaskAllocationStrategy):
    def allocate_tasks(self, unassigned_tasks: List[Task], idle_robots: List[RobotState]) -> Dict[str, str]:
        allocations = {}
        robot_idx = 0
        for task in unassigned_tasks:
            if robot_idx < len(idle_robots):
                allocations[task.task_id] = idle_robots[robot_idx].robot_id
                robot_idx += 1
            else:
                break
        return allocations
