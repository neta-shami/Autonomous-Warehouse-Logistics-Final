"""Derive conservative grid exclusions from one immutable fleet snapshot."""

from math import hypot
from typing import FrozenSet, Iterable, Set, Tuple

from entities.enums import FSMStatus
from entities.fleet_snapshot import FleetSnapshot
from entities.warehouse_topology import WarehouseTopology


def blocked_cells_for_robot(
    snapshot: FleetSnapshot,
    topology: WarehouseTopology,
    moving_robot_id: str,
    moving_radius_m: float,
    physical_margin_m: float = 0.05,
) -> FrozenSet[Tuple[int, int]]:
    """Expand every known disabled footprint into hard grid obstacles."""
    disabled_ids = {
        robot.robot_id
        for robot in snapshot.robots
        if robot.active_faults or robot.fsm_status is FSMStatus.FAULTED
    }
    return blocked_cells_for_robot_ids(
        snapshot,
        topology,
        moving_robot_id,
        moving_radius_m,
        disabled_ids,
        physical_margin_m,
    )


def blocked_cells_for_robot_ids(
    snapshot: FleetSnapshot,
    topology: WarehouseTopology,
    moving_robot_id: str,
    moving_radius_m: float,
    blocked_robot_ids: Iterable[str],
    physical_margin_m: float = 0.05,
) -> FrozenSet[Tuple[int, int]]:
    """Expand selected robot footprints into conservative grid exclusions."""
    blocked = set()
    selected: Set[str] = set(blocked_robot_ids)
    for robot in snapshot.robots:
        if robot.robot_id == moving_robot_id or robot.pose_estimate is None:
            continue
        if robot.robot_id not in selected:
            continue
        expanded_radius = (
            robot.footprint_radius_m
            + robot.pose_estimate.uncertainty_m
            + moving_radius_m
            + physical_margin_m
        )
        center = robot.pose_estimate.pose
        for x in range(topology.grid_width):
            for y in range(topology.grid_height):
                if hypot(center.x - x, center.y - y) <= expanded_radius:
                    blocked.add((x, y))
    return frozenset(blocked)
