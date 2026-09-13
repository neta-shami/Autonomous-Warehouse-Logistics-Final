"""Owned, versioned inventory reservation for one task attempt."""

from dataclasses import dataclass
from typing import Optional, Tuple

from entities.value_validation import require_nonnegative_integer


@dataclass
class TaskLease:
    task_id: str
    lease_generation: int
    payload_id: Optional[str]
    source_slot: Optional[Tuple[int, int, int]]
    target_slot: Tuple[int, int, int]
    source_released: bool = False
    target_released: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string")
        require_nonnegative_integer(
            "lease_generation", self.lease_generation
        )
        if self.lease_generation == 0:
            raise ValueError("lease_generation must be positive")
        if self.payload_id is not None and (
            not isinstance(self.payload_id, str) or not self.payload_id
        ):
            raise ValueError("payload_id must be a non-empty string or None")
        for name, slot in (
            ("source_slot", self.source_slot),
            ("target_slot", self.target_slot),
        ):
            if slot is not None and (
                not isinstance(slot, tuple)
                or len(slot) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in slot
                )
            ):
                raise TypeError(f"{name} must be an integer x/y/tier tuple")
        if not isinstance(self.source_released, bool) or not isinstance(
            self.target_released, bool
        ):
            raise TypeError("lease release flags must be bool")
