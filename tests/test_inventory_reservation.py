import unittest

from entities.enums import SnapshotType, TaskType
from entities.events import PackageDroppedEvent, PackagePickedEvent
from entities.inventory_world import InventoryWorld
from entities.task import Task
from tests.topology_support import sample_topology
from use_cases.inventory_reservation_service import InventoryReservationService


class TestInventoryReservationService(unittest.TestCase):
    def setUp(self):
        self.inventory_world = InventoryWorld()
        self.service = InventoryReservationService(
            self.inventory_world, sample_topology()
        )

    def test_rejects_endpoint_outside_the_physical_topology(self):
        task = Task.transfer(
            "invalid", TaskType.RELOCATE, (50, 50), 0, (4, 4), 0, "p1"
        )

        with self.assertRaisesRegex(ValueError, "outside the warehouse grid"):
            self.service.process_new_task(task, 1.0)

        self.assertEqual(self.service.ready_queue, [])
        self.assertEqual(self.inventory_world.task_leases, {})

    def test_rejects_wrong_zone_for_task_type(self):
        task = Task.transfer(
            "invalid", TaskType.RETRIEVE, (2, 2), 0, (4, 4), 0, "p1"
        )

        with self.assertRaisesRegex(ValueError, "OUTBOUND_DOCK"):
            self.service.process_new_task(task, 1.0)

    def test_store_task_blocked_if_no_package_spawned(self):
        # A STORE task expects a package physically at the source (INBOUND_DOCK)
        task = Task.transfer("t1", TaskType.STORE, (9, 1), 0, (2, 2), 1)
        self.service.process_new_task(task, 1.0)

        self.assertEqual(len(self.service.blocked_queue), 1)
        self.assertEqual(len(self.service.ready_queue), 0)

        # Now simulate scenario manager spawning the package
        physical = self.inventory_world.get_snapshot(SnapshotType.PHYSICAL)
        physical.set_cell_occupancy(9, 1, 0, "pkg1")

        # Trigger re-evaluation
        self.service.evaluate_blocked_tasks(1.5)

        self.assertEqual(len(self.service.blocked_queue), 0)
        self.assertEqual(len(self.service.ready_queue), 1)

    def test_retrieve_task_blocked_if_outbound_dock_full(self):
        # A RETRIEVE task expects the OUTBOUND_DOCK to be physically empty
        task = Task.transfer("t2", TaskType.RETRIEVE, (2, 2), 1, (9, 8), 0, "p2")

        # Simulate a package currently sitting at the outbound dock waiting to despawn
        physical = self.inventory_world.get_snapshot(SnapshotType.PHYSICAL)
        physical.set_cell_occupancy(9, 8, 0, "old_pkg")

        # Simulate physical presence at source
        physical.set_cell_occupancy(2, 2, 1, "p2")

        self.service.process_new_task(task, 2.0)

        self.assertEqual(len(self.service.blocked_queue), 1)
        self.assertEqual(len(self.service.ready_queue), 0)

        # Simulate despawn
        physical.set_cell_occupancy(9, 8, 0, False)
        self.service.evaluate_blocked_tasks(2.5)

        self.assertEqual(len(self.service.blocked_queue), 0)
        self.assertEqual(len(self.service.ready_queue), 1)

    def test_events_update_physical_world(self):
        task = Task.transfer("t3", TaskType.RELOCATE, (2, 2), 1, (4, 4), 1, "pkg3")

        # Manually put package in physical world
        physical = self.inventory_world.get_snapshot(SnapshotType.PHYSICAL)
        physical.set_cell_occupancy(2, 2, 1, "pkg3")
        self.service.process_new_task(task, 2.5)

        # Robot picks package
        pick_event = PackagePickedEvent(
            sim_time=3.0,
            robot_id="r1",
            task=task,
            lease_generation=task.lease_generation,
        )
        self.service.handle_package_picked(pick_event)

        # Verify physical world is empty at source
        self.assertTrue(physical.is_cell_empty(2, 2, 1))

        # Robot drops package
        drop_event = PackageDroppedEvent(
            sim_time=3.5,
            robot_id="r1",
            task=task,
            lease_generation=task.lease_generation,
        )
        self.service.handle_package_dropped(drop_event)

        # Verify physical world has package at target
        self.assertFalse(physical.is_cell_empty(4, 4, 1))

    def test_relocate_infers_payload_identity_from_physical_source(self):
        physical = self.inventory_world.get_snapshot(SnapshotType.PHYSICAL)
        physical.set_cell_occupancy(2, 2, 0, "p7")
        task = Task.transfer(
            "move", TaskType.RELOCATE, (2, 2), 0, (4, 4), 0
        )

        self.service.process_new_task(task, 1.0)

        self.assertEqual(task.payload_id, "p7")
        self.assertIn(task, self.service.ready_queue)

    def test_duplicate_admission_is_idempotent_for_the_same_task(self):
        physical = self.inventory_world.get_snapshot(SnapshotType.PHYSICAL)
        physical.set_cell_occupancy(2, 2, 0, "p7")
        task = Task.transfer(
            "move", TaskType.RELOCATE, (2, 2), 0, (4, 4), 0, "p7"
        )

        self.service.process_new_task(task, 1.0)
        self.service.process_new_task(task, 1.1)

        self.assertEqual(self.service.ready_queue, [task])
        self.assertEqual(self.service.blocked_queue, [])

    def test_explicit_payload_mismatch_is_rejected_without_reservation(self):
        physical = self.inventory_world.get_snapshot(SnapshotType.PHYSICAL)
        reserved = self.inventory_world.get_snapshot(SnapshotType.RESERVED)
        physical.set_cell_occupancy(2, 2, 0, "actual")
        task = Task.transfer(
            "move", TaskType.RELOCATE, (2, 2), 0, (4, 4), 0, "wrong"
        )

        with self.assertRaisesRegex(ValueError, "actual"):
            self.service.process_new_task(task, 1.0)

        self.assertTrue(reserved.is_cell_empty(4, 4, 0))
        self.assertNotIn(task, self.service.ready_queue)

if __name__ == '__main__':
    unittest.main()
