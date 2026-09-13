"""Transactional ownership and event-version tests for inventory leases."""

from entities.enums import SnapshotType, TaskStatus, TaskType
from entities.events import PackageDroppedEvent, PackagePickedEvent, TaskCompletedEvent
from entities.inventory_world import InventoryWorld
from entities.task import Task
from tests.topology_support import sample_topology
from use_cases.inventory_reservation_service import InventoryReservationService


def _service_with_payloads(*entries):
    world = InventoryWorld()
    physical = world.get_snapshot(SnapshotType.PHYSICAL)
    for coords, tier, payload_id in entries:
        physical.set_cell_occupancy(*coords, tier, payload_id)
    return world, InventoryReservationService(
        world, sample_topology((1, 1), (1, 2))
    )


def _move(task_id, source, target, payload_id=None):
    return Task.transfer(
        task_id, TaskType.RELOCATE, source, 0, target, 0, payload_id
    )


def test_competing_source_or_target_acquires_neither_endpoint():
    world, service = _service_with_payloads(
        ((1, 1), 0, "p1"), ((1, 2), 0, "p2")
    )
    first = _move("first", (1, 1), (5, 5), "p1")
    same_source = _move("same-source", (1, 1), (6, 6), "p1")
    same_target = _move("same-target", (1, 2), (5, 5), "p2")

    service.process_new_task(first, 1.0)
    service.process_new_task(same_source, 1.1)
    service.process_new_task(same_target, 1.2)

    assert set(world.task_leases) == {"first"}
    assert service.blocked_queue == [same_source, same_target]
    projected = world.get_snapshot(SnapshotType.RESERVED)
    assert projected.is_cell_empty(6, 6, 0)


def test_stale_and_duplicate_events_cannot_mutate_current_attempt():
    world, service = _service_with_payloads(((1, 1), 0, "p1"))
    task = _move("move", (1, 1), (5, 5), "p1")
    service.process_new_task(task, 1.0)
    old_generation = task.lease_generation
    service.ready_queue.clear()
    old_failure = TaskCompletedEvent(
        1.1,
        "move",
        "r1",
        False,
        task,
        lease_generation=old_generation,
    )
    service.handle_task_completed(old_failure)
    service.process_new_task(task, 1.2)
    assert task.lease_generation == old_generation + 1

    service.handle_task_completed(old_failure)
    assert world.task_leases[task.task_id].lease_generation == old_generation + 1
    assert (
        world.get_snapshot(SnapshotType.RESERVED).get_cell_occupancy(5, 5, 0)
        == "p1"
    )

    stale = PackagePickedEvent(1.3, "r1", task, old_generation)
    service.handle_package_picked(stale)
    physical = world.get_snapshot(SnapshotType.PHYSICAL)
    assert physical.get_cell_occupancy(1, 1, 0) == "p1"

    current = PackagePickedEvent(
        1.4, "r1", task, task.lease_generation
    )
    service.handle_package_picked(current)
    service.handle_package_picked(current)
    assert physical.is_cell_empty(1, 1, 0)


def test_success_releases_all_ownership_and_duplicate_drop_is_idempotent():
    world, service = _service_with_payloads(((1, 1), 0, "p1"))
    task = _move("move", (1, 1), (5, 5), "p1")
    service.process_new_task(task, 1.0)
    generation = task.lease_generation
    service.handle_package_picked(
        PackagePickedEvent(1.1, "r1", task, generation)
    )
    drop = PackageDroppedEvent(1.2, "r1", task, generation)
    service.handle_package_dropped(drop)
    service.handle_package_dropped(drop)

    assert world.get_snapshot(SnapshotType.PHYSICAL).get_cell_occupancy(
        5, 5, 0
    ) == "p1"
    assert world.task_leases == {}


def test_loaded_recovery_task_retains_target_lease():
    world, service = _service_with_payloads(((1, 1), 0, "p1"))
    task = _move("move", (1, 1), (5, 5), "p1")
    service.process_new_task(task, 1.0)
    service.handle_package_picked(
        PackagePickedEvent(1.1, "r1", task, task.lease_generation)
    )

    task.update_status(TaskStatus.RECOVERY_REQUIRED)

    lease = world.task_leases[task.task_id]
    assert lease.source_released
    assert not lease.target_released
    assert lease.target_slot == (5, 5, 0)
