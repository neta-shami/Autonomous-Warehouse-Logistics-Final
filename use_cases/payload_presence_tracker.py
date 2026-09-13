"""Single debounce authority for the gripper's raw payload ray."""

import math
from typing import Optional

from entities.payload_presence_assessment import PayloadPresenceAssessment
from entities.value_validation import require_finite, require_nonnegative_integer


class PayloadPresenceTracker:
    """Require committed consecutive near/no-hit samples before changing state."""

    def __init__(
        self,
        confirmation_samples: int = 2,
        present_max_range_m: float = 0.08,
    ):
        require_nonnegative_integer("confirmation_samples", confirmation_samples)
        if confirmation_samples == 0:
            raise ValueError("confirmation_samples must be at least one")
        require_finite(
            "present_max_range_m", present_max_range_m, minimum=0.0
        )
        self.confirmation_samples = confirmation_samples
        self.present_max_range_m = present_max_range_m
        self._candidate: Optional[bool] = None
        self._consecutive_samples = 0
        self._stable_presence: Optional[bool] = None

    @property
    def stable_presence(self) -> Optional[bool]:
        return self._stable_presence

    def assess(self, payload_range: Optional[float]) -> PayloadPresenceAssessment:
        """Classify raw range and propose state without mutating history."""
        sample = self._classify(payload_range)
        if sample is None:
            return PayloadPresenceAssessment(
                None,
                None,
                self._candidate,
                self._consecutive_samples,
                False,
            )
        count = (
            self._consecutive_samples + 1
            if sample is self._candidate
            else 1
        )
        stable = (
            sample
            if count >= self.confirmation_samples
            else self._stable_presence
        )
        return PayloadPresenceAssessment(sample, stable, sample, count, True)

    def commit(self, assessment: PayloadPresenceAssessment) -> None:
        """Advance debounce history only for the accepted tick assessment."""
        if not isinstance(assessment, PayloadPresenceAssessment):
            raise TypeError("assessment must be PayloadPresenceAssessment")
        if not assessment.advance:
            return
        self._candidate = assessment.candidate
        self._consecutive_samples = assessment.consecutive_samples
        self._stable_presence = assessment.payload_present

    def _classify(self, payload_range: Optional[float]) -> Optional[bool]:
        if payload_range is None or isinstance(payload_range, bool):
            return None
        try:
            finite = math.isfinite(payload_range)
        except TypeError:
            return None
        if not finite:
            return None
        if payload_range == -1.0:
            return False
        if 0.0 <= payload_range <= self.present_max_range_m:
            return True
        return None
