"""General snapshot-based detour policy for non-head-on traffic stalls."""

from interfaces.i_traffic_resolution_strategy import ITrafficResolutionStrategy
from use_cases.traffic_strategies.routing import try_snapshot_detour


class SnapshotDetourStrategy(ITrafficResolutionStrategy):
    """Replan around one or several blockers using the committed snapshot."""

    def __init__(self, topology, path_planner):
        self.topology = topology
        self.path_planner = path_planner

    def resolve(self, event, robots, snapshot, blocked_cells=frozenset()):
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
