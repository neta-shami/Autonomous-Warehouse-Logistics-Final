"""
Warehouse Scenario Manager Module

Role: Manages high-level scenario rules, such as package arrivals (spawning) and departures (despawning).
Acts as the bridge between the logical inventory system and the physical MuJoCo environment.
"""
from dataclasses import dataclass
from typing import List, Set, Tuple

from entities.enums import SnapshotType, TaskType
from entities.events import TaskCompletedEvent
from entities.task import Task
from entities.value_validation import require_finite
from interfaces.i_environment import IEnvironment
from use_cases.inventory_reservation_service import InventoryReservationService


@dataclass(frozen=True)
class ScheduledDespawn:
    due_sim_time: float
    coords: Tuple[int, int]
    tier: int
    task_id: str
    payload_id: str


class WarehouseScenarioManager:
    """
    Coordinates timed environmental actions (like despawning retrieved packages).
    """
    def __init__(self, environment: IEnvironment, reservation_service: InventoryReservationService):
        self.environment = environment
        self.reservation_service = reservation_service
        self.despawn_timers: List[ScheduledDespawn] = []
        self._handled_retrieve_attempts: Set[Tuple[str, int]] = set()

    def tick(self, sim_time: float) -> None:
        """Processes time-based scenario events like despawning."""
        require_finite("scenario tick sim_time", sim_time, minimum=0.0)
        for timer in self.despawn_timers[:]:
            if sim_time < timer.due_sim_time:
                continue
            outbound_x, outbound_y = timer.coords
            physical = self.reservation_service.inventory_world.get_snapshot(
                SnapshotType.PHYSICAL
            )
            if physical.get_cell_occupancy(
                outbound_x, outbound_y, timer.tier
            ) != timer.payload_id:
                raise RuntimeError(
                    "Outbound physical inventory does not match scheduled payload"
                )
            if not self.environment.despawn_package(
                timer.payload_id, outbound_x, outbound_y, radius=1.0
            ):
                continue
            physical.set_cell_occupancy(outbound_x, outbound_y, timer.tier, False)
            projected = self.reservation_service.inventory_world.get_snapshot(
                SnapshotType.RESERVED
            )
            projected.set_cell_occupancy(
                outbound_x, outbound_y, timer.tier, False
            )
            self.despawn_timers.remove(timer)
            self.reservation_service.evaluate_blocked_tasks(sim_time)

    def spawn_package(
        self,
        package_id: str,
        target_shelf: Tuple[int, int],
        target_tier: int,
        sim_time: float,
    ) -> bool:
        """
        Attempts to spawn a package at the inbound dock and assigns a storage task.
        """
        require_finite("spawn command sim_time", sim_time, minimum=0.0)
        inbound_x, inbound_y = (9, 1)
        physical = self.reservation_service.inventory_world.get_snapshot(SnapshotType.PHYSICAL)
        if not physical.is_cell_empty(inbound_x, inbound_y, 0):
            print("Spawn rejected: INBOUND_DOCK occupied")
            return False

        spawned_id = self.environment.spawn_package_physically(
            package_id, inbound_x, inbound_y, 0.46
        )
        if not spawned_id:
            print("Spawn rejected: No packages available in environment")
            return False

        physical.set_cell_occupancy(inbound_x, inbound_y, 0, package_id)

        t = Task.transfer(
            task_id=package_id,
            task_type=TaskType.STORE,
            source_coords=(inbound_x, inbound_y),
            source_tier=0,
            target_coords=target_shelf,
            target_tier=target_tier,
            payload_id=package_id,
        )
        self.reservation_service.process_new_task(t, sim_time)
        self.reservation_service.evaluate_blocked_tasks(sim_time)
        return True

    def schedule_despawn(
        self,
        delay: float,
        coords: Tuple[int, int],
        tier: int,
        task_id: str,
        payload_id: str,
        sim_time: float,
    ) -> None:
        """Schedules a package despawn in the future."""
        require_finite("despawn delay", delay, minimum=0.0)
        require_finite("despawn sim_time", sim_time, minimum=0.0)
        if not isinstance(payload_id, str) or not payload_id:
            raise ValueError("payload_id must be a non-empty string")
        scheduled = ScheduledDespawn(
            due_sim_time=sim_time + delay,
            coords=coords,
            tier=tier,
            task_id=task_id,
            payload_id=payload_id,
        )
        for existing in self.despawn_timers:
            if existing.task_id != task_id:
                continue
            if existing == scheduled:
                return
            raise ValueError(
                f"Task '{task_id}' already has a different despawn schedule"
            )
        self.despawn_timers.append(scheduled)

    def handle_task_completed(self, event: TaskCompletedEvent) -> None:
        """
        [Event Listener]
        Called when any robot finishes a task.
        If the task was a RETRIEVE, it schedules the package to be despawned
        (simulating a truck taking it away from the Outbound Dock).
        """
        if (
            event.success
            and event.task
            and event.task.task_type == TaskType.RETRIEVE
        ):
            attempt = (event.task_id, event.lease_generation)
            if attempt in self._handled_retrieve_attempts:
                return
            if event.task.payload_id is None:
                raise ValueError("Completed RETRIEVE task must identify its payload")
            self.schedule_despawn(
                delay=3.0,
                coords=event.task.target_coords,
                tier=event.task.target_tier,
                task_id=event.task.task_id,
                payload_id=event.task.payload_id,
                sim_time=event.sim_time,
            )
            self._handled_retrieve_attempts.add(attempt)
