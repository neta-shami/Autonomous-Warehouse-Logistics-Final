"""Commit-gated proposal for payload presence debounce state."""

from dataclasses import dataclass
from typing import Optional

from entities.value_validation import (
    require_nonnegative_integer,
    validate_optional_bool,
)


@dataclass(frozen=True)
class PayloadPresenceAssessment:
    """Proposed raw classification and next debounce state."""

    payload_sample: Optional[bool]
    payload_present: Optional[bool]
    candidate: Optional[bool]
    consecutive_samples: int
    advance: bool

    def __post_init__(self) -> None:
        validate_optional_bool("payload_sample", self.payload_sample)
        validate_optional_bool("payload_present", self.payload_present)
        validate_optional_bool("candidate", self.candidate)
        require_nonnegative_integer(
            "consecutive_samples", self.consecutive_samples
        )
        if not isinstance(self.advance, bool):
            raise TypeError("advance must be bool")
