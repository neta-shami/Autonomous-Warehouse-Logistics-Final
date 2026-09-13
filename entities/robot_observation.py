from dataclasses import dataclass
from typing import Optional, Tuple

from entities.value_validation import (
    require_finite,
    require_nonnegative_integer,
    validate_optional_bool,
    validate_optional_velocity,
)


@dataclass(frozen=True)
class RobotObservation:
    """Same-tick non-pose sensor interpretation awaiting atomic commit."""

    observed_chassis_velocity: Optional[Tuple[float, float]]
    payload_sample: Optional[bool]
    payload_present: Optional[bool]
    sample_sim_time: Optional[float]
    sequence: Optional[int]

    def __post_init__(self) -> None:
        validate_optional_velocity(self.observed_chassis_velocity)
        validate_optional_bool("payload_sample", self.payload_sample)
        validate_optional_bool("payload_present", self.payload_present)
        if (self.sample_sim_time is None) != (self.sequence is None):
            raise ValueError("sample_sim_time and sequence must be provided together")
        if self.sample_sim_time is not None:
            require_finite("sample_sim_time", self.sample_sim_time, minimum=0.0)
        if self.sequence is not None:
            require_nonnegative_integer("sequence", self.sequence)
