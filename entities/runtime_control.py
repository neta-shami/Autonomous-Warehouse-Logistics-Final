"""Immutable proposals and command context for an atomic logic tick."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, FrozenSet, Optional, Tuple

from entities.enums import CommandGateReason
from entities.fleet_snapshot import FleetSnapshot
from entities.health import HealthFinding
from entities.localization_assessment import LocalizationAssessment
from entities.payload_presence_assessment import PayloadPresenceAssessment
from entities.robot_observation import RobotObservation
from entities.sensor_frame import SensorFrame
from entities.value_validation import require_finite

if TYPE_CHECKING:
    from entities.events import WarehouseEvent


@dataclass(frozen=True)
class RobotTickAssessment:
    """All side-effect-free proposals prepared for one robot and one tick."""

    localization: LocalizationAssessment
    observation: RobotObservation
    payload_presence: PayloadPresenceAssessment
    health_findings: Tuple[HealthFinding, ...]
    stop_required: bool
    events: Tuple["WarehouseEvent", ...]

    def __post_init__(self) -> None:
        if not isinstance(self.localization, LocalizationAssessment):
            raise TypeError("localization must be LocalizationAssessment")
        if not isinstance(self.observation, RobotObservation):
            raise TypeError("observation must be RobotObservation")
        if not isinstance(self.payload_presence, PayloadPresenceAssessment):
            raise TypeError("payload_presence must be PayloadPresenceAssessment")
        if not isinstance(self.health_findings, tuple):
            raise TypeError("health_findings must be an immutable tuple")
        if not isinstance(self.stop_required, bool):
            raise TypeError("stop_required must be bool")
        if not isinstance(self.events, tuple):
            raise TypeError("events must be an immutable tuple")


@dataclass(frozen=True)
class TickContext:
    """Same snapshot, timestamp, and safety gates supplied to one robot act."""

    frame: Optional[SensorFrame]
    fleet_snapshot: FleetSnapshot
    sim_time: float
    dt: float
    gate_reasons: FrozenSet[CommandGateReason]

    def __post_init__(self) -> None:
        if self.frame is not None and not isinstance(self.frame, SensorFrame):
            raise TypeError("frame must be SensorFrame or None")
        if not isinstance(self.fleet_snapshot, FleetSnapshot):
            raise TypeError("fleet_snapshot must be FleetSnapshot")
        require_finite("tick sim_time", self.sim_time, minimum=0.0)
        require_finite("tick dt", self.dt, minimum=0.0)
        if not isinstance(self.gate_reasons, frozenset) or any(
            not isinstance(reason, CommandGateReason)
            for reason in self.gate_reasons
        ):
            raise TypeError("gate_reasons must be a frozenset of CommandGateReason")

    @property
    def commands_allowed(self) -> bool:
        return not self.gate_reasons
