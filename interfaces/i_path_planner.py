"""Port for continuous-to-grid warehouse route planning."""

from abc import ABC, abstractmethod
from typing import FrozenSet, List, Optional, Sequence, Tuple, Union

from entities.pose import Pose
from entities.warehouse_topology import WarehouseTopology


class IPathPlanner(ABC):
    """Plan routes without exposing a concrete search implementation."""

    @abstractmethod
    def find_path(
        self,
        start: Union[Pose, Tuple[int, int], Tuple[float, float]],
        goal: Tuple[int, int],
        topology: WarehouseTopology,
        active_paths: Sequence[Sequence[Tuple[int, int]]],
        blocked_cells: FrozenSet[Tuple[int, int]] = frozenset(),
        footprint_radius_m: float = 0.0,
    ) -> List[Tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def find_path_to_access(
        self,
        start: Union[Pose, Tuple[int, int], Tuple[float, float]],
        target: Tuple[int, int],
        topology: WarehouseTopology,
        active_paths: Sequence[Sequence[Tuple[int, int]]],
        blocked_cells: FrozenSet[Tuple[int, int]] = frozenset(),
        footprint_radius_m: float = 0.0,
    ) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
        """Return the shortest reachable manipulation access and its route."""
        raise NotImplementedError
