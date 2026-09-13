"""Collision, command-gate, and inventory-integrity metrics."""

from math import hypot
from typing import Iterable, Tuple

from entities.drive_command import DriveCommand
from entities.inventory_world import InventoryWorld
from entities.value_validation import require_finite


class SafetyMetrics:
    """Own measurements proving that unsafe physical actions did not occur."""

    def __init__(self) -> None:
        self.near_collision_stops = 0
        self.total_actual_collisions = 0
        self.unsafe_command_batches = 0
        self.orphan_lease_count = 0
        self.peak_orphan_lease_count = 0

    def record_near_collision_stop(self) -> None:
        self.near_collision_stops += 1

    def record_robot_collisions(
        self, collision_pairs: Iterable[Tuple[str, str]]
    ) -> None:
        self.total_actual_collisions += len(set(collision_pairs))

    def record_command_result(
        self,
        sim_time: float,
        *,
        unsafe: bool,
        command: DriveCommand,
    ) -> None:
        require_finite("command result sim_time", sim_time, minimum=0.0)
        if not isinstance(command, DriveCommand):
            raise TypeError("command must be DriveCommand")
        if unsafe and hypot(command.vx, command.vy) > 1e-12:
            self.unsafe_command_batches += 1

    def sample_orphan_leases(
        self,
        inventory_world: InventoryWorld,
        active_task_ids: Iterable[str],
    ) -> None:
        active = set(active_task_ids)
        self.orphan_lease_count = sum(
            task_id not in active for task_id in inventory_world.task_leases
        )
        self.peak_orphan_lease_count = max(
            self.peak_orphan_lease_count,
            self.orphan_lease_count,
        )
