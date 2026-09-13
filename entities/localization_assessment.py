from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

from entities.absolute_pose_fix import AbsolutePoseFix
from entities.enums import LocalizationStatus
from entities.pose_estimate import PoseEstimate
from entities.value_validation import (
    require_finite,
    require_nonnegative_integer,
)

if TYPE_CHECKING:
    from entities.events import WarehouseEvent


@dataclass(frozen=True)
class ConfirmationUpdate:
    """Proposed bounded confirmation state for a candidate absolute fix."""

    candidate: Optional[AbsolutePoseFix]
    consecutive_samples: int
    confirmed: bool

    def __post_init__(self) -> None:
        if self.candidate is not None and not isinstance(
            self.candidate, AbsolutePoseFix
        ):
            raise TypeError("candidate must be an AbsolutePoseFix or None")
        require_nonnegative_integer(
            "consecutive_samples", self.consecutive_samples
        )
        if not isinstance(self.confirmed, bool):
            raise TypeError("confirmed must be bool")
        if self.candidate is None and (
            self.consecutive_samples != 0 or self.confirmed
        ):
            raise ValueError("confirmation samples require a candidate")
        if self.candidate is not None and self.consecutive_samples == 0:
            raise ValueError("a candidate requires at least one sample")


@dataclass(frozen=True)
class LocalizationAssessment:
    """Side-effect-free localization proposal for the atomic commit phase."""

    estimate: Optional[PoseEstimate]
    status: LocalizationStatus
    stop_required: bool
    divergence_m: Optional[float]
    confirmation: ConfirmationUpdate
    events: Tuple["WarehouseEvent", ...]

    def __post_init__(self) -> None:
        if self.estimate is not None and not isinstance(
            self.estimate, PoseEstimate
        ):
            raise TypeError("estimate must be a PoseEstimate or None")
        if not isinstance(self.status, LocalizationStatus):
            raise TypeError("status must be LocalizationStatus")
        if not isinstance(self.stop_required, bool):
            raise TypeError("stop_required must be bool")
        if self.divergence_m is not None:
            require_finite("divergence_m", self.divergence_m, minimum=0.0)
        if not isinstance(self.confirmation, ConfirmationUpdate):
            raise TypeError("confirmation must be ConfirmationUpdate")
        if not isinstance(self.events, tuple):
            raise TypeError("events must be an immutable tuple")
