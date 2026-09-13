"""Head-on detour and safe retry policy."""

from interfaces.i_traffic_resolution_strategy import ITrafficResolutionStrategy
from use_cases.traffic_strategies.routing import try_snapshot_detour


class HeadOnStrategy(ITrafficResolutionStrategy):
    def __init__(self, topology, path_planner):
        self.topology = topology
        self.path_planner = path_planner

    def resolve(self, event, robots, snapshot, blocked_cells=frozenset()):
        agent = robots.get(event.robot_id)
        if agent is None or agent.current_task is None:
            return
        if try_snapshot_detour(
            event,
            agent,
            snapshot,
            blocked_cells,
            self.topology,
            self.path_planner,
        ):
            return

        # No route in one snapshot is a temporary traffic condition, not a
        # failed warehouse task. FleetManager retries against fresh positions.
        agent.route_replan_required = True
