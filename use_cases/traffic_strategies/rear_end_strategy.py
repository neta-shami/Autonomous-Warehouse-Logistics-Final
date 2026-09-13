"""Rear-end traffic detour policy."""

from interfaces.i_traffic_resolution_strategy import ITrafficResolutionStrategy
from use_cases.traffic_strategies.routing import try_snapshot_detour


class RearEndStrategy(ITrafficResolutionStrategy):
    def __init__(self, topology, path_planner):
        self.topology = topology
        self.path_planner = path_planner

    def resolve(self, event, robots, snapshot, blocked_cells=frozenset()):
        if not event.blocking_robot_ids:
            return
        agent = robots.get(event.robot_id)
        if agent is not None:
            try_snapshot_detour(
                event,
                agent,
                snapshot,
                blocked_cells,
                self.topology,
                self.path_planner,
            )
