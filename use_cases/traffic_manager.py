"""
Traffic Manager Module

Role: Responsible for preventing and resolving robot gridlock.
It monitors the grid for collisions and intercepts navigation
when an obstacle is detected.
"""

import logging
import math
from typing import Dict, Optional, Tuple

from entities.enums import TrafficPattern
from entities.events import CollisionStallEvent
from entities.fleet_snapshot import FleetRobotView, FleetSnapshot
from entities.warehouse_topology import WarehouseTopology
from interfaces.i_path_planner import IPathPlanner
from interfaces.i_robot_registry import IRobotRegistry

# Strategy interfaces
from interfaces.i_traffic_resolution_strategy import ITrafficResolutionStrategy
from use_cases.dynamic_obstacles import (
    blocked_cells_for_robot,
    blocked_cells_for_robot_ids,
)

LOGGER = logging.getLogger(__name__)


class TrafficManager:
    """
    [Pattern: Observer / Pub-Sub (Subscriber)]
    The TrafficManager subscribes to the EventDispatcher to automatically
    receive CollisionStallEvents, eliminating manual loops in the Orchestrator.

    [Pattern: Context (Strategy Pattern)]
    Acts as the Context for resolving traffic stalls. It categorizes the geometric
    pattern of the cluster, and dispatches a specific ITrafficResolutionStrategy
    (e.g., HeadOnStrategy, RearEndStrategy).
    """

    def __init__(
        self,
        topology: WarehouseTopology,
        path_planner: IPathPlanner,
        robot_registry: IRobotRegistry,
    ):
        self.topology = topology
        self.path_planner = path_planner

        # We will load these strategies later. For now, they can be None or simple stubs.
        self.strategies: Dict[TrafficPattern, ITrafficResolutionStrategy] = {}
        self.robots = robot_registry
        self.latest_snapshot: Optional[FleetSnapshot] = None

    def update_snapshot(self, snapshot: FleetSnapshot) -> None:
        """Receive the committed fleet view used for this logic tick."""
        self.latest_snapshot = snapshot

    def set_strategy(self, pattern: TrafficPattern, strategy: ITrafficResolutionStrategy) -> None:
        self.strategies[pattern] = strategy

    def handle_stall_event(self, event: CollisionStallEvent) -> None:
        """
        [Event Listener]
        Called automatically when a NavigatingState robot detects it is stuck.
        Calculates the angle of intersection between the two robots, categorizes
        the deadlock (e.g. HEAD_ON, REAR_END), and executes a traffic resolution strategy.
        """
        if not tuple(self.robots.values()):
            return

        agent = self.robots.get(event.robot_id)
        if not agent or not agent.current_task:
            return

        if self.latest_snapshot is None:
            return
        pattern = self._categorize_pattern(event, self.latest_snapshot)

        strategy = self.strategies.get(pattern)
        if strategy:
            view = self.latest_snapshot.robot(event.robot_id)
            blocked_cells = blocked_cells_for_robot(
                self.latest_snapshot,
                self.topology,
                event.robot_id,
                view.footprint_radius_m,
            )
            # A healthy idle or yielding robot is still a hard obstacle for
            # this detour. Without this, the planner repeatedly selected the
            # same occupied cell and recreated the stall.
            blocked_cells = blocked_cells.union(
                blocked_cells_for_robot_ids(
                    self.latest_snapshot,
                    self.topology,
                    event.robot_id,
                    view.footprint_radius_m,
                    event.blocking_robot_ids,
                )
            )
            strategy.resolve(
                event, self.robots, self.latest_snapshot, blocked_cells
            )
        else:
            LOGGER.warning(
                "No strategy for %s; robot %s remains safely stopped",
                pattern.name,
                event.robot_id,
            )

    def _categorize_pattern(
        self, event: CollisionStallEvent, snapshot: FleetSnapshot
    ) -> TrafficPattern:
        """Determine stall geometry from the committed same-tick snapshot."""
        blocking_ids = event.blocking_robot_ids

        if not blocking_ids:
            return TrafficPattern.UNKNOWN

        if len(blocking_ids) > 1:
            return TrafficPattern.STUCK_BETWEEN

        primary_blocker_id = blocking_ids[0]
        try:
            blocker = snapshot.robot(primary_blocker_id)
            agent = snapshot.robot(event.robot_id)
        except KeyError:
            return TrafficPattern.UNKNOWN

        # Calculate heading vectors (assuming heading is in radians)
        # Note: If heading is not available in radians, we can infer from path
        # Let's infer from current pos to next waypoint if possible
        v1 = self._get_intent_vector(agent)
        v2 = self._get_intent_vector(blocker)

        if v1 == (0,0) or v2 == (0,0):
            return TrafficPattern.REAR_END # One is parked

        # Dot product
        dot = v1[0]*v2[0] + v1[1]*v2[1]

        if dot < -0.5:
            return TrafficPattern.HEAD_ON
        elif dot > 0.5:
            return TrafficPattern.REAR_END
        else:
            return TrafficPattern.CROSSING

    def _get_intent_vector(
        self, robot: FleetRobotView
    ) -> Tuple[float, float]:
        if robot.pose_estimate is None or not robot.active_path:
            return (0.0, 0.0)

        target = robot.active_path[0]
        curr = robot.pose_estimate.pose
        dx = target[0] - curr.x
        dy = target[1] - curr.y

        length = math.hypot(dx, dy)
        if length == 0:
            return (0.0, 0.0)

        return (dx/length, dy/length)
