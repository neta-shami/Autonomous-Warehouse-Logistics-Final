"""Immutable, same-tick fleet state shared by every robot action."""

from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple

from entities.enums import FaultCode, FSMStatus, LocalizationStatus
from entities.pose_estimate import PoseEstimate
from entities.value_validation import (
    require_finite,
    require_nonnegative_integer,
    validate_optional_velocity,
)


@dataclass(frozen=True)
class FleetRobotView:
    """One robot's bounded committed state inside a fleet snapshot."""

    robot_id: str
    pose_estimate: Optional[PoseEstimate]
    observed_chassis_velocity: Optional[Tuple[float, float]]
    footprint_radius_m: float
    active_path: Tuple[Tuple[int, int], ...]
    localization_status: LocalizationStatus
    active_faults: FrozenSet[FaultCode]
    fsm_status: FSMStatus

    def __post_init__(self) -> None:
        if not isinstance(self.robot_id, str) or not self.robot_id:
            raise ValueError("robot_id must be a non-empty string")
        if self.pose_estimate is not None and not isinstance(
            self.pose_estimate, PoseEstimate
        ):
            raise TypeError("pose_estimate must be PoseEstimate or None")
        validate_optional_velocity(self.observed_chassis_velocity)
        require_finite(
            "footprint_radius_m", self.footprint_radius_m, minimum=0.0
        )
        if not isinstance(self.active_path, tuple):
            raise TypeError("active_path must be an immutable tuple")
        for waypoint in self.active_path:
            if not isinstance(waypoint, tuple) or len(waypoint) != 2:
                raise TypeError("active_path waypoints must be x/y tuples")
        if not isinstance(self.localization_status, LocalizationStatus):
            raise TypeError("localization_status must be LocalizationStatus")
        if not isinstance(self.active_faults, frozenset) or any(
            not isinstance(fault, FaultCode) for fault in self.active_faults
        ):
            raise TypeError("active_faults must be a frozenset of FaultCode")
        if not isinstance(self.fsm_status, FSMStatus):
            raise TypeError("fsm_status must be FSMStatus")


@dataclass(frozen=True)
class FleetSnapshot:
    """One versioned fleet view used unchanged by all acts in a tick."""

    version: int
    sim_time: float
    robots: Tuple[FleetRobotView, ...]

    def __post_init__(self) -> None:
        require_nonnegative_integer("snapshot version", self.version)
        require_finite("snapshot sim_time", self.sim_time, minimum=0.0)
        if not isinstance(self.robots, tuple):
            raise TypeError("robots must be an immutable tuple")
        ids = [robot.robot_id for robot in self.robots]
        if len(ids) != len(set(ids)):
            raise ValueError("snapshot robot ids must be unique")

    def robot(self, robot_id: str) -> FleetRobotView:
        """Return one robot view or fail explicitly for a wiring error."""
        for robot in self.robots:
            if robot.robot_id == robot_id:
                return robot
        raise KeyError(f"Robot '{robot_id}' is absent from fleet snapshot")
