from abc import ABC, abstractmethod
from typing import FrozenSet, Sequence, Tuple

from entities.warehouse_topology import WarehouseTopology


class IPathfindingStrategy(ABC):
    @abstractmethod
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        topology: WarehouseTopology,
        active_paths: Sequence[Sequence[Tuple[int, int]]],
        blocked_cells: FrozenSet[Tuple[int, int]] = frozenset(),
    ) -> list[Tuple[int, int]]:
        """
        Calculates a path from start to goal.
        Penalizes or avoids cells in active_paths to prevent collisions.
        Returns a list of (x, y) tuples representing the path.
        """
        raise NotImplementedError
