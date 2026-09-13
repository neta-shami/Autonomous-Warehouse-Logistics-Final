"""Identity-safe outbound despawn and retry behavior."""

from unittest.mock import MagicMock

import pytest

from entities.enums import SnapshotType, TaskType
from entities.events import TaskCompletedEvent
from entities.inventory_world import InventoryWorld
from entities.task import Task
from tests.topology_support import sample_topology
from use_cases.inventory_reservation_service import InventoryReservationService
from use_cases.warehouse_scenario_manager import WarehouseScenarioManager


def _retrieve_task() -> Task:
    return Task.transfer(
        "retrieve-p1", TaskType.RETRIEVE, (2, 5), 0, (9, 8), 0, "p1"
    )


def test_failed_retrieve_never_schedules_a_despawn():
    environment = MagicMock()
    manager = WarehouseScenarioManager(
        environment,
        InventoryReservationService(InventoryWorld(), sample_topology()),
    )
    task = _retrieve_task()

    manager.handle_task_completed(
        TaskCompletedEvent(1.0, task.task_id, "r1", False, task)
    )

    assert manager.despawn_timers == []


def test_outbound_despawn_targets_identity_and_retries_until_success():
    environment = MagicMock()
    environment.despawn_package.return_value = False
    world = InventoryWorld()
    service = InventoryReservationService(world, sample_topology())
    manager = WarehouseScenarioManager(environment, service)
    task = _retrieve_task()
    physical = world.get_snapshot(SnapshotType.PHYSICAL)
    reserved = world.get_snapshot(SnapshotType.RESERVED)
    physical.set_cell_occupancy(9, 8, 0, "p1")
    reserved.set_cell_occupancy(9, 8, 0, "p1")
    completed = TaskCompletedEvent(1.0, task.task_id, "r1", True, task)
    manager.handle_task_completed(completed)

    manager.tick(4.0)

    environment.despawn_package.assert_called_once_with("p1", 9, 8, radius=1.0)
    assert len(manager.despawn_timers) == 1
    assert physical.get_cell_occupancy(9, 8, 0) == "p1"

    environment.despawn_package.return_value = True
    manager.tick(4.02)

    assert manager.despawn_timers == []
    assert physical.is_cell_empty(9, 8, 0)
    assert reserved.is_cell_empty(9, 8, 0)

    manager.handle_task_completed(completed)
    assert manager.despawn_timers == []


def test_duplicate_completion_schedules_only_one_despawn():
    manager = WarehouseScenarioManager(
        MagicMock(),
        InventoryReservationService(InventoryWorld(), sample_topology()),
    )
    task = _retrieve_task()
    event = TaskCompletedEvent(1.0, task.task_id, "r1", True, task)

    manager.handle_task_completed(event)
    manager.handle_task_completed(event)

    assert len(manager.despawn_timers) == 1


def test_inventory_identity_is_checked_before_physical_despawn():
    environment = MagicMock()
    world = InventoryWorld()
    manager = WarehouseScenarioManager(
        environment, InventoryReservationService(world, sample_topology())
    )
    task = _retrieve_task()
    world.get_snapshot(SnapshotType.PHYSICAL).set_cell_occupancy(
        9, 8, 0, "different-payload"
    )
    manager.handle_task_completed(
        TaskCompletedEvent(1.0, task.task_id, "r1", True, task)
    )

    with pytest.raises(RuntimeError, match="scheduled payload"):
        manager.tick(4.0)
    environment.despawn_package.assert_not_called()
