"""Single owner of the live robot collection."""

from typing import Iterable, Optional, Tuple

from interfaces.i_robot_registry import ICoordinatedRobot, IRobotRegistry


class RobotRegistry(IRobotRegistry):
    """Expose robot capabilities through a read-only registry port."""

    def __init__(self) -> None:
        self._robots: dict[str, ICoordinatedRobot] = {}

    def get(self, robot_id: str) -> Optional[ICoordinatedRobot]:
        return self._robots.get(robot_id)

    def values(self) -> Iterable[ICoordinatedRobot]:
        return tuple(self._robots[key] for key in sorted(self._robots))

    def items(self) -> Iterable[Tuple[str, ICoordinatedRobot]]:
        return tuple((key, self._robots[key]) for key in sorted(self._robots))

    def robot_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._robots))

    def register(self, robot_id: str, robot: ICoordinatedRobot) -> None:
        if not isinstance(robot_id, str) or not robot_id:
            raise ValueError("robot_id must be a non-empty string")
        if robot_id in self._robots:
            raise ValueError(f"Robot '{robot_id}' is already registered")
        self._robots[robot_id] = robot

    def replace(self, robots: Iterable[Tuple[str, ICoordinatedRobot]]) -> None:
        items = tuple(robots)
        replacement = dict(items)
        if len(replacement) != len(items):
            raise ValueError("robot ids must be unique")
        self._robots = replacement
