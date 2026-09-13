"""
Inventory Reservation Service Module

Role: Acts as the gatekeeper for all tasks in the warehouse.
It ensures that two robots never try to pick up or drop off at the same shelf at the same time.
It maintains logical snapshots of the inventory (Physical and Reserved).
"""
from typing import List

from entities.enums import SnapshotType, TaskType
from entities.events import (
    PackageDroppedEvent,
    PackagePickedEvent,
    TaskCompletedEvent,
    TaskReadyEvent,
)
from entities.inventory_world import InventoryWorld
from entities.task import Task
from entities.value_validation import require_finite
from entities.warehouse_topology import WarehouseTopology


class InventoryReservationService:
    """
    Manages the task queues (blocked vs ready) and maintains logical consistency
    of the warehouse grid by preventing conflicting task assignments.
    """
    def __init__(
        self,
        inventory_world: InventoryWorld,
        topology: WarehouseTopology,
    ):
        if not isinstance(inventory_world, InventoryWorld):
            raise TypeError("inventory_world must be an InventoryWorld")
        if not isinstance(topology, WarehouseTopology):
            raise TypeError("topology must be a WarehouseTopology")
        self.inventory_world = inventory_world
        self.topology = topology
        self.blocked_queue: List[Task] = []
        self.ready_queue: List[Task] = []
        self.event_dispatcher = None

    def set_event_dispatcher(self, dispatcher) -> None:
        self.event_dispatcher = dispatcher

    def process_new_task(self, task: Task, sim_time: float) -> None:
        """
        [Event Publisher]
        Validates a new task against the physical and reserved grids.
        If valid, it adds it to the ready_queue and publishes a TaskReadyEvent.
        If blocked, it goes to the blocked_queue.
        """
        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        require_finite("task command sim_time", sim_time, minimum=0.0)
        self.topology.validate_task_locations(task)
        for queued in self.ready_queue + self.blocked_queue:
            if queued.task_id != task.task_id:
                continue
            if queued is task:
                return
            raise ValueError(f"Task id '{task.task_id}' is already queued")
        active_lease = self.inventory_world.task_leases.get(task.task_id)
        if active_lease is not None:
            if active_lease.lease_generation == task.lease_generation:
                return
            raise ValueError(f"Task id '{task.task_id}' already owns a lease")
        reserved_snapshot = self.inventory_world.get_snapshot(SnapshotType.RESERVED)
        physical = self.inventory_world.get_snapshot(SnapshotType.PHYSICAL)

        # 1. Target must be logically free
        if not reserved_snapshot.is_cell_empty(task.target_coords[0], task.target_coords[1], task.target_tier):
            self.blocked_queue.append(task)
            return

        # 2. For RETRIEVE tasks, the OUTBOUND_DOCK must be physically empty (no pending despawns)
        if task.task_type == TaskType.RETRIEVE and not physical.is_cell_empty(
            task.target_coords[0], task.target_coords[1], task.target_tier
        ):
            self.blocked_queue.append(task)
            return

        # 3. For STORE, RETRIEVE, and RELOCATE, the package must be physically present at the source!
        if task.task_type in (TaskType.STORE, TaskType.RETRIEVE, TaskType.RELOCATE):
            source_coords = task.source_coords
            source_tier = task.source_tier
            if source_coords is None or source_tier is None:
                raise ValueError("transfer task requires a complete source slot")
            if physical.is_cell_empty(*source_coords, source_tier):
                self.blocked_queue.append(task)
                return
            source_payload = physical.get_cell_occupancy(
                *source_coords, source_tier
            )
            if not isinstance(source_payload, str):
                raise ValueError("transfer source occupancy must identify a payload")
            if task.payload_id is None:
                task.payload_id = source_payload
            elif task.payload_id != source_payload:
                raise ValueError(
                    f"Task '{task.task_id}' expects payload '{task.payload_id}', "
                    f"but source contains '{source_payload}'"
                )

        lease = self.inventory_world.acquire_task_lease(task)
        if lease is None:
            self.blocked_queue.append(task)
            return

        # Project the payload into its destination; ownership is kept separately.
        reserved_snapshot.set_cell_occupancy(
            task.target_coords[0], task.target_coords[1], task.target_tier,
            task.payload_id or task.task_id,
        )
        self.ready_queue.append(task)

        if self.event_dispatcher:
            self.event_dispatcher.dispatch(
                TaskReadyEvent(sim_time=sim_time)
            )

    def evaluate_blocked_tasks(self, sim_time: float) -> None:
        require_finite("task retry sim_time", sim_time, minimum=0.0)
        to_process = self.blocked_queue[:]
        self.blocked_queue.clear()
        for task in to_process:
            self.process_new_task(task, sim_time)

    def handle_package_picked(self, event: PackagePickedEvent) -> None:
        """
        [Event Listener]
        Called when a robot's arm physically grabs a box.
        Removes the package from the physical grid snapshot.
        """
        physical = self.inventory_world.get_snapshot(SnapshotType.PHYSICAL)
        lease = self.inventory_world.task_leases.get(event.task.task_id)
        if (
            lease is None
            or lease.lease_generation != event.lease_generation
            or lease.source_released
            or lease.source_slot is None
            or not isinstance(lease.payload_id, str)
        ):
            return
        occupant = physical.get_cell_occupancy(*lease.source_slot)
        if occupant != lease.payload_id:
            return
        physical.set_cell_occupancy(*lease.source_slot, False)
        self.inventory_world.get_snapshot(
            SnapshotType.RESERVED
        ).set_cell_occupancy(*lease.source_slot, False)
        self.inventory_world.release_source(
            event.task.task_id, event.lease_generation
        )
        self.evaluate_blocked_tasks(event.sim_time)

    def handle_package_dropped(self, event: PackageDroppedEvent) -> None:
        """
        [Event Listener]
        Called when a robot's arm drops a box onto a shelf.
        Adds the package back into the physical grid snapshot.
        """
        physical = self.inventory_world.get_snapshot(SnapshotType.PHYSICAL)
        lease = self.inventory_world.task_leases.get(event.task.task_id)
        if (
            lease is None
            or lease.lease_generation != event.lease_generation
            or lease.target_released
            or not isinstance(lease.payload_id, str)
            or not physical.is_cell_empty(*lease.target_slot)
        ):
            return
        physical.set_cell_occupancy(*lease.target_slot, lease.payload_id)
        self.inventory_world.get_snapshot(
            SnapshotType.RESERVED
        ).set_cell_occupancy(*lease.target_slot, lease.payload_id)
        self.inventory_world.release_target(
            event.task.task_id, event.lease_generation
        )
        self.evaluate_blocked_tasks(event.sim_time)

    def handle_task_completed(self, event: TaskCompletedEvent) -> None:
        """
        [Event Listener]
        Called when a robot finishes a task. Frees up the reserved cell.
        """
        if event.task and not event.success:
            lease = self.inventory_world.task_leases.get(event.task.task_id)
            if (
                lease is not None
                and lease.lease_generation == event.lease_generation
            ):
                target_slot = lease.target_slot
                self.inventory_world.release_remaining(
                    event.task.task_id, event.lease_generation
                )
                self.inventory_world.get_snapshot(
                    SnapshotType.RESERVED
                ).set_cell_occupancy(*target_slot, False)
        self.evaluate_blocked_tasks(event.sim_time)
