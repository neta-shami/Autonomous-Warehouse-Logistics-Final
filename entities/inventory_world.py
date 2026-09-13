from typing import Dict, Optional, Tuple

from .enums import SnapshotType
from .grid_state_snapshot import GridStateSnapshot
from .task import Task
from .task_lease import TaskLease


class InventoryWorld:
    def __init__(self):
        self.snapshots: Dict[SnapshotType, GridStateSnapshot] = {
            SnapshotType.PHYSICAL: GridStateSnapshot(),
            SnapshotType.RESERVED: GridStateSnapshot(),
        }
        self.task_leases: Dict[str, TaskLease] = {}
        self._slot_owners: Dict[Tuple[int, int, int], Tuple[str, int]] = {}
        self._last_generation: Dict[str, int] = {}

    def get_snapshot(self, state_type: SnapshotType) -> GridStateSnapshot:
        return self.snapshots[state_type]

    def acquire_task_lease(self, task: Task) -> Optional[TaskLease]:
        """Acquire source and target together, changing nothing on conflict."""
        if task.task_id in self.task_leases:
            return None
        if task.source_coords is None:
            source = None
        else:
            if task.source_tier is None:
                raise ValueError("task source coordinates require a source tier")
            source = (*task.source_coords, task.source_tier)
        target = (*task.target_coords, task.target_tier)
        slots = tuple(slot for slot in (source, target) if slot is not None)
        if len(set(slots)) != len(slots) or any(
            slot in self._slot_owners for slot in slots
        ):
            return None
        generation = self._last_generation.get(task.task_id, 0) + 1
        lease = TaskLease(
            task.task_id, generation, task.payload_id, source, target
        )
        for slot in slots:
            self._slot_owners[slot] = (task.task_id, generation)
        self._last_generation[task.task_id] = generation
        self.task_leases[task.task_id] = lease
        task.lease_generation = generation
        return lease

    def release_source(self, task_id: str, generation: int) -> bool:
        return self._release(task_id, generation, source=True)

    def release_target(self, task_id: str, generation: int) -> bool:
        return self._release(task_id, generation, source=False)

    def release_remaining(self, task_id: str, generation: int) -> bool:
        """Release both endpoints still owned by a terminal attempt."""
        lease = self.task_leases.get(task_id)
        if lease is None or lease.lease_generation != generation:
            return False
        self._release(task_id, generation, source=True)
        self._release(task_id, generation, source=False)
        return True

    def _release(self, task_id: str, generation: int, *, source: bool) -> bool:
        lease = self.task_leases.get(task_id)
        if lease is None or lease.lease_generation != generation:
            return False
        slot = lease.source_slot if source else lease.target_slot
        already_released = (
            lease.source_released if source else lease.target_released
        )
        if already_released:
            return True
        if slot is not None and self._slot_owners.get(slot) != (task_id, generation):
            return False
        if slot is not None:
            self._slot_owners.pop(slot, None)
        if source:
            lease.source_released = True
        else:
            lease.target_released = True
        if lease.source_released and lease.target_released:
            self.task_leases.pop(task_id, None)
        return True
