"""Robot belief state; simulator ground truth never belongs here."""

from dataclasses import dataclass
from typing import FrozenSet, Optional, Tuple

from entities.enums import FaultCode, LocalizationStatus
from entities.pose_estimate import PoseEstimate
from entities.value_validation import (
    validate_optional_bool,
    validate_optional_velocity,
)


@dataclass
class RobotState:
    """The robot's committed pose, observation, payload, and fault beliefs."""

    robot_id: str
    pose_estimate: Optional[PoseEstimate]
    observed_chassis_velocity: Optional[Tuple[float, float]]
    expected_payload_id: Optional[str]
    payload_present: Optional[bool]
    is_busy: bool
    home_base_coords: Tuple[int, int]
    localization_status: LocalizationStatus = LocalizationStatus.UNINITIALIZED
    active_faults: FrozenSet[FaultCode] = frozenset()
    last_observation_sim_time: Optional[float] = None
    last_observation_sequence: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.robot_id, str) or not self.robot_id:
            raise ValueError("robot_id must be a non-empty string")
        if self.pose_estimate is not None and not isinstance(
            self.pose_estimate, PoseEstimate
        ):
            raise TypeError("pose_estimate must be PoseEstimate or None")
        validate_optional_velocity(self.observed_chassis_velocity)
        if self.expected_payload_id is not None and (
            not isinstance(self.expected_payload_id, str)
            or not self.expected_payload_id
        ):
            raise ValueError("expected_payload_id must be a non-empty string or None")
        validate_optional_bool("payload_present", self.payload_present)
        if not isinstance(self.is_busy, bool):
            raise TypeError("is_busy must be bool")
        if (
            not isinstance(self.home_base_coords, tuple)
            or len(self.home_base_coords) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.home_base_coords
            )
        ):
            raise TypeError("home_base_coords must be an integer x/y tuple")
        if not isinstance(self.localization_status, LocalizationStatus):
            raise TypeError("localization_status must be LocalizationStatus")
        if not isinstance(self.active_faults, frozenset) or any(
            not isinstance(fault, FaultCode) for fault in self.active_faults
        ):
            raise TypeError("active_faults must be a frozenset of FaultCode")
        self._validate_localization_pair(
            self.pose_estimate, self.localization_status
        )

    @property
    def current_coords(self) -> Tuple[float, float]:
        """Return believed coordinates, failing loudly while pose is unknown."""
        if self.pose_estimate is None:
            raise RuntimeError(f"Robot '{self.robot_id}' has no usable pose")
        return (self.pose_estimate.pose.x, self.pose_estimate.pose.y)

    def apply_localization(
        self,
        estimate: Optional[PoseEstimate],
        status: LocalizationStatus,
    ) -> None:
        """Atomically replace the accepted pose and localization status."""
        if estimate is not None and not isinstance(estimate, PoseEstimate):
            raise TypeError("estimate must be PoseEstimate or None")
        if not isinstance(status, LocalizationStatus):
            raise TypeError("status must be LocalizationStatus")
        self._validate_localization_pair(estimate, status)
        self.pose_estimate = estimate
        self.localization_status = status

    def apply_observation(
        self,
        observed_chassis_velocity: Optional[Tuple[float, float]],
        payload_present: Optional[bool],
        sample_sim_time: Optional[float] = None,
        sequence: Optional[int] = None,
    ) -> None:
        """Commit same-frame velocity and debounced payload observations."""
        validate_optional_velocity(observed_chassis_velocity)
        validate_optional_bool("payload_present", payload_present)
        if (sample_sim_time is None) != (sequence is None):
            raise ValueError("sample_sim_time and sequence must be provided together")
        if sample_sim_time is not None:
            from entities.value_validation import (
                require_finite,
                require_nonnegative_integer,
            )

            require_finite("sample_sim_time", sample_sim_time, minimum=0.0)
            if sequence is None:
                raise ValueError("sequence is required with sample_sim_time")
            require_nonnegative_integer("sequence", sequence)
        self.observed_chassis_velocity = observed_chassis_velocity
        self.payload_present = payload_present
        self.last_observation_sim_time = sample_sim_time
        self.last_observation_sequence = sequence

    def activate_faults(self, *faults: FaultCode) -> None:
        """Atomically add validated independent fault codes."""
        if any(not isinstance(fault, FaultCode) for fault in faults):
            raise TypeError("faults must contain only FaultCode values")
        self.active_faults = frozenset(set(self.active_faults).union(faults))

    def replace_faults(self, faults: FrozenSet[FaultCode]) -> None:
        """Replace the committed fault snapshot during the atomic commit phase."""
        if not isinstance(faults, frozenset) or any(
            not isinstance(fault, FaultCode) for fault in faults
        ):
            raise TypeError("faults must be a frozenset of FaultCode")
        self.active_faults = faults

    def clear_fault(self, fault: FaultCode) -> bool:
        """Remove one active fault; policy authorization belongs to the agent."""
        if not isinstance(fault, FaultCode):
            raise TypeError("fault must be FaultCode")
        if fault not in self.active_faults:
            return False
        self.active_faults = frozenset(
            active for active in self.active_faults if active is not fault
        )
        return True

    @staticmethod
    def _validate_localization_pair(
        estimate: Optional[PoseEstimate], status: LocalizationStatus
    ) -> None:
        usable = status in (
            LocalizationStatus.TRUSTED,
            LocalizationStatus.DEGRADED,
        )
        if usable != (estimate is not None):
            raise ValueError(
                "localization status and pose estimate are incoherent"
            )
