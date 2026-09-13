from __future__ import annotations

import uuid
from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from entities.enums import FaultCode
from entities.pose import Pose
from entities.task import Task
from entities.value_validation import require_finite, require_nonnegative_integer


@dataclass
class WarehouseEvent(ABC):
    """Base event stamped explicitly with the publisher's simulation time."""

    sim_time: float
    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        require_finite("event sim_time", self.sim_time, minimum=0.0)
        self.event_id = uuid.uuid4().hex

@dataclass
class CollisionStallEvent(WarehouseEvent):
    """
    Fired when a robot is unable to move due to an obstacle or traffic jam.
    Publisher: NavigatingState (when check_collisions returns AVOIDING).
    Subscriber: TrafficManager (calculates traffic resolution strategies).
    """
    robot_id: str
    coords: Tuple[float, float]
    blocking_robot_ids: List[str]

@dataclass
class TaskCompletedEvent(WarehouseEvent):
    """
    Fired when a robot successfully (or unsuccessfully) finishes a warehouse task.
    Publisher: Task strategies (e.g., TransferTaskStrategy, ParkTaskStrategy).
    Subscribers:
    - InventoryReservationService (updates logical package statuses)
    - WarehouseScenarioManager (triggers despawn for outbound packages)
    - MetricsCollector (increments total system throughput)
    """
    task_id: str
    robot_id: str
    success: bool
    task: Optional[Task] = None
    lease_generation: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        require_nonnegative_integer("lease_generation", self.lease_generation)

@dataclass
class PackagePickedEvent(WarehouseEvent):
    """
    Fired when a robot's manipulator successfully picks up a package from a shelf/dock.
    Publisher: ManipulatingState.
    Subscriber: InventoryReservationService (removes the package from the physical grid snapshot).
    """
    robot_id: str
    task: Task
    lease_generation: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        require_nonnegative_integer("lease_generation", self.lease_generation)

@dataclass
class PackageDroppedEvent(WarehouseEvent):
    """
    Fired when a robot's manipulator successfully drops a package onto a shelf/dock.
    Publisher: ManipulatingState.
    Subscriber: InventoryReservationService (adds the package back into the physical grid snapshot).
    """
    robot_id: str
    task: Task
    lease_generation: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        require_nonnegative_integer("lease_generation", self.lease_generation)

@dataclass
class RobotIdleEvent(WarehouseEvent):
    """
    Fired exactly once when a robot finishes all active tasks and enters the idle state.
    Publisher: IdleState.
    Subscriber: FleetManager (dispatches queued tasks or auto-parks the robot).
    """
    robot_id: str
    current_coords: Tuple[int, int]

@dataclass
class TaskReadyEvent(WarehouseEvent):
    """
    Fired when a task successfully passes all inventory checks and enters the ready queue.
    Publisher: InventoryReservationService.
    Subscriber: FleetManager (triggers task allocation algorithm instantly).
    """
    pass


@dataclass
class RelocalizedEvent(WarehouseEvent):
    """Reports that a stopped robot has established a new trusted pose."""

    robot_id: str
    recovered: Pose
    recovery_duration_s: float


@dataclass
class RelocalizationFailedEvent(WarehouseEvent):
    """Reports that localization could not recover inside its deadline."""

    robot_id: str


@dataclass
class PayloadMismatchEvent(WarehouseEvent):
    """Reports a stable payload-presence mismatch for the current task phase."""

    robot_id: str


@dataclass
class RobotFaultedEvent(WarehouseEvent):
    """Reports which robot entered its committed faulted state."""

    robot_id: str


@dataclass
class FaultResetEvent(WarehouseEvent):
    """Auditable notification that evidence cleared one latched fault."""

    robot_id: str
    cleared_fault: FaultCode
