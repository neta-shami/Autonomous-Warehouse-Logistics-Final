import heapq
from typing import FrozenSet, Sequence, Set, Tuple

from entities.warehouse_topology import WarehouseTopology
from interfaces.i_pathfinding_strategy import IPathfindingStrategy


class AStarPathfindingStrategy(IPathfindingStrategy):
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        topology: WarehouseTopology,
        active_paths: Sequence[Sequence[Tuple[int, int]]],
        blocked_cells: FrozenSet[Tuple[int, int]] = frozenset(),
    ) -> list[Tuple[int, int]]:
        if (
            not topology.is_navigable(*start)
            or not topology.is_navigable(*goal)
            or start in blocked_cells
            or goal in blocked_cells
        ):
            return []
        occupied_cells: Set[Tuple[int, int]] = set()
        for path in active_paths:
            for cell in path:
                occupied_cells.add(cell)

        open_set: list[tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, start))
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if (
                    not topology.is_navigable(*neighbor)
                    or neighbor in blocked_cells
                ):
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor in occupied_cells:
                    tentative_g_score += 100

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(
        self, came_from: dict, current: Tuple[int, int]
    ) -> list[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
