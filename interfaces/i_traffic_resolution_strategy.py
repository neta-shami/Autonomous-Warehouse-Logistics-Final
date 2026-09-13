"""Interface for traffic resolution strategies.

Each strategy encapsulates a specific way to resolve a particular traffic pattern.
"""

from abc import ABC, abstractmethod
from typing import FrozenSet, Tuple

from entities.events import CollisionStallEvent
from entities.fleet_snapshot import FleetSnapshot
from interfaces.i_robot_registry import IRobotRegistry


class ITrafficResolutionStrategy(ABC):
    """Strategy interface for resolving specific traffic patterns (e.g., Rear-End, Crossing).

    Implementations should orchestrate the maneuvers needed to untangle the robots.
    """

    @abstractmethod
    def resolve(
        self,
        event: CollisionStallEvent,
        robots: IRobotRegistry,
        snapshot: FleetSnapshot,
        blocked_cells: FrozenSet[Tuple[int, int]] = frozenset(),
    ) -> None:
        """Resolve the traffic situation.

        Args:
            event: The stall event containing the robot ID and the blocking robot IDs.
            robots: Port for resolving the affected robot by ID.
            snapshot: Immutable same-tick fleet state.
            blocked_cells: Hard exclusions derived from disabled robots.
        """
        raise NotImplementedError
