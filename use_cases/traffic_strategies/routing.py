"""Shared snapshot-only traffic detour construction."""

from typing import FrozenSet, Tuple

from entities.events import CollisionStallEvent
from entities.fleet_snapshot import FleetSnapshot
from entities.warehouse_topology import WarehouseTopology
from interfaces.i_path_planner import IPathPlanner
from interfaces.i_robot_registry import ICoordinatedRobot


def try_snapshot_detour(
    event: CollisionStallEvent,
    agent: ICoordinatedRobot,
    snapshot: FleetSnapshot,
    blocked_cells: FrozenSet[Tuple[int, int]],
    topology: WarehouseTopology,
    path_planner: IPathPlanner,
) -> bool:
    """Apply a detour derived from a committed view and the preserved leg."""
    if agent.current_task is None or agent.current_leg_goal is None:
        return False
    view = snapshot.robot(event.robot_id)
    if view.pose_estimate is None:
        return False
    start = view.pose_estimate.pose
    active_paths = tuple(
        robot.active_path
        for robot in snapshot.robots
        if robot.robot_id != event.robot_id
    )
    path = path_planner.find_path(
        start,
        agent.current_leg_goal,
        topology,
        active_paths,
        blocked_cells=blocked_cells,
        footprint_radius_m=agent.FOOTPRINT_RADIUS_M,
    )
    if not path:
        return False
    agent.set_path(path)
    return True
