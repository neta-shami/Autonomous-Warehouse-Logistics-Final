from enum import Enum, auto
from math import pi


class ZoneType(Enum):
    INBOUND_DOCK = auto()
    OUTBOUND_DOCK = auto()
    NORMAL_FLOOR = auto()
    SHELF = auto()
    PARKING = auto()


class CardinalDirection(Enum):
    """A world-aligned direction shared by routing and robot hardware."""

    POS_X = auto()
    NEG_X = auto()
    POS_Y = auto()
    NEG_Y = auto()

    @property
    def angle_radians(self) -> float:
        """Return the world-aligned turret angle for this direction."""
        return {
            CardinalDirection.POS_X: 0.0,
            CardinalDirection.NEG_X: pi,
            CardinalDirection.POS_Y: pi / 2.0,
            CardinalDirection.NEG_Y: -pi / 2.0,
        }[self]


class ManipulatorResult(Enum):
    IN_PROGRESS = auto()
    SUCCEEDED = auto()
    FAILED = auto()


class ManipulatorAbortResult(Enum):
    CLEAN = auto()
    RECONCILIATION_REQUIRED = auto()


class PoseSource(Enum):
    """How a position estimate was produced."""

    PREDICTED = auto()
    ABSOLUTE_FIX = auto()
    FUSED = auto()


class LocalizationStatus(Enum):
    UNINITIALIZED = auto()
    TRUSTED = auto()
    DEGRADED = auto()
    LOST = auto()


class FaultCode(Enum):
    DRIVE_STALL = auto()
    PAYLOAD_MISMATCH = auto()
    SENSOR_INVALID = auto()
    STATE_TIMEOUT = auto()
    MANIPULATOR_FAILURE = auto()
    POSITION_PHYSICALLY_INVALID = auto()


class FaultSeverity(Enum):
    STOP = auto()
    QUARANTINE = auto()


class CommandGateReason(Enum):
    LOCAL_SAFETY = auto()
    FLEET_HOLD = auto()

class TaskType(Enum):
    STORE = auto()
    RETRIEVE = auto()
    RELOCATE = auto()
    PARK = auto()

class TaskStatus(Enum):
    PENDING = auto()
    QUEUED = auto()
    IN_PROGRESS = auto()
    RECOVERY_REQUIRED = auto()
    COMPLETED = auto()
    FAILED = auto()


class TaskExecutionPhase(Enum):
    TO_SOURCE = auto()
    PICKING = auto()
    TO_TARGET = auto()
    DROPPING = auto()
    COMPLETE = auto()

class FSMStatus(Enum):
    IDLE = auto()
    NAVIGATING = auto()
    AVOIDING = auto()
    MANIPULATING = auto()
    RELOCALIZING = auto()
    FAULTED = auto()

class SnapshotType(Enum):
    PHYSICAL = auto()
    RESERVED = auto()

class TrafficPattern(Enum):
    REAR_END = auto()
    HEAD_ON = auto()
    CROSSING = auto()
    STUCK_BETWEEN = auto()
    UNKNOWN = auto()
