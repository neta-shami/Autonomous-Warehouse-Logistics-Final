from math import hypot
from typing import FrozenSet, List, Optional, Sequence, Tuple, Union

from entities.pose import Pose
from entities.warehouse_topology import WarehouseTopology
from interfaces.i_path_planner import IPathPlanner
from interfaces.i_pathfinding_strategy import IPathfindingStrategy


class GridPathPlanner(IPathPlanner):
    """Adapt a continuous robot pose to safe grid-based A* routing."""

    MAX_CONNECTOR_DISTANCE_M = 1.5

    def __init__(self, strategy: IPathfindingStrategy):
        self.strategy = strategy

    def find_path(
        self,
        start: Union[Pose, Tuple[int, int], Tuple[float, float]],
        goal: Tuple[int, int],
        topology: WarehouseTopology,
        active_paths: Sequence[Sequence[Tuple[int, int]]],
        blocked_cells: FrozenSet[Tuple[int, int]] = frozenset(),
        footprint_radius_m: float = 0.0,
    ) -> list[Tuple[int, int]]:
        if not isinstance(blocked_cells, frozenset):
            raise TypeError("blocked_cells must be a frozenset")
        start_pose = self._as_pose(start)
        anchor = self._select_start_anchor(
            start_pose, topology, blocked_cells, footprint_radius_m
        )
        if anchor is None:
            return []
        footprint_blocked = frozenset(
            (x, y)
            for x in range(topology.grid_width)
            for y in range(topology.grid_height)
            if not topology.is_pose_feasible(
                Pose(float(x), float(y)), footprint_radius_m
            )
        )
        return self.strategy.find_path(
            anchor,
            goal,
            topology,
            active_paths,
            blocked_cells | footprint_blocked,
        )

    def find_path_to_access(
        self,
        start: Union[Pose, Tuple[int, int], Tuple[float, float]],
        target: Tuple[int, int],
        topology: WarehouseTopology,
        active_paths: Sequence[Sequence[Tuple[int, int]]],
        blocked_cells: FrozenSet[Tuple[int, int]] = frozenset(),
        footprint_radius_m: float = 0.0,
    ) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
        """Choose the shortest currently reachable access around a station."""
        start_pose = self._as_pose(start)
        robot_grid = (int(round(start_pose.x)), int(round(start_pose.y)))
        reachable = []
        for access in topology.get_access_points(target, robot_grid):
            path = self.find_path(
                start_pose,
                access,
                topology,
                active_paths,
                blocked_cells=blocked_cells,
                footprint_radius_m=footprint_radius_m,
            )
            if path:
                reachable.append((len(path), access, path))
        if not reachable:
            return None, []
        _, access, path = min(reachable, key=lambda item: (item[0], item[1]))
        return access, path

    def _select_start_anchor(
        self,
        start: Pose,
        topology: WarehouseTopology,
        blocked_cells: FrozenSet[Tuple[int, int]],
        footprint_radius_m: float,
    ) -> Optional[Tuple[int, int]]:
        if not topology.is_pose_feasible(start, footprint_radius_m):
            return None
        candidates = []
        for x in range(topology.grid_width):
            for y in range(topology.grid_height):
                anchor = (x, y)
                distance = hypot(start.x - x, start.y - y)
                if (
                    distance <= self.MAX_CONNECTOR_DISTANCE_M
                    and anchor not in blocked_cells
                    and topology.is_navigable(x, y)
                ):
                    candidates.append((distance, anchor))
        for _, anchor in sorted(candidates, key=lambda item: (item[0], item[1])):
            anchor_pose = Pose(float(anchor[0]), float(anchor[1]))
            if topology.is_segment_feasible(
                start, anchor_pose, footprint_radius_m
            ):
                return anchor
        return None

    @staticmethod
    def _as_pose(
        start: Union[Pose, Tuple[int, int], Tuple[float, float]]
    ) -> Pose:
        if isinstance(start, Pose):
            return start
        if not isinstance(start, tuple) or len(start) != 2:
            raise TypeError("start must be Pose or an x/y tuple")
        return Pose(start[0], start[1])
