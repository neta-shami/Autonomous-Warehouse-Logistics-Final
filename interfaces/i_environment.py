from abc import abstractmethod
from typing import Iterable, Optional, Set, Tuple

from interfaces.i_simulation_clock import ISimulationClock


class IEnvironment(ISimulationClock):
    """
    Abstract interface for the physics environment.
    Decouples the logic orchestrator from the underlying physics engine (e.g., MuJoCo).
    """

    @abstractmethod
    def step(self) -> None:
        """Advances the physics simulation by one step."""
        pass

    @abstractmethod
    def get_time(self) -> float:
        """Returns the current simulation time."""
        pass

    @abstractmethod
    def get_robot_collisions(
        self, robot_ids: Iterable[str]
    ) -> Set[Tuple[str, str]]:
        """Return unique unintended robot collision pairs for this tick."""
        pass

    @abstractmethod
    def spawn_package_physically(
        self, package_id: str, x: float, y: float, z: float
    ) -> Optional[str]:
        """
        Spawn the named package at the given coordinates, or return None when
        that physical package is missing or already active.
        """
        pass

    @abstractmethod
    def despawn_package(
        self, payload_id: str, x: float, y: float, radius: float
    ) -> bool:
        """
        Remove the expected package only when it is near the requested location.
        """
        pass
