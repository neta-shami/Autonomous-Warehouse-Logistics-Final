"""Metrics for explicitly injected robot displacement and recovery."""

from typing import Dict, FrozenSet

from entities.events import RelocalizationFailedEvent, RelocalizedEvent
from entities.value_validation import require_finite


class RecoveryMetrics:
    """Pair every injected discontinuity with exactly one recovery outcome."""

    def __init__(self) -> None:
        self.total_external_discontinuities = 0
        self.recovered_external_discontinuities = 0
        self.external_recovery_duration_s = 0.0
        self.false_relocations = 0
        self._expected_relocations: Dict[str, int] = {}

    @property
    def pending_robot_ids(self) -> FrozenSet[str]:
        return frozenset(
            robot_id for robot_id, count in self._expected_relocations.items() if count
        )

    def mark_external_discontinuity(self, robot_id: str, sim_time: float) -> None:
        if not isinstance(robot_id, str) or not robot_id:
            raise ValueError("robot_id must be a non-empty string")
        require_finite("discontinuity sim_time", sim_time, minimum=0.0)
        self._expected_relocations[robot_id] = (
            self._expected_relocations.get(robot_id, 0) + 1
        )
        self.total_external_discontinuities += 1

    def handle_relocalized(self, event: RelocalizedEvent) -> None:
        pending = self._expected_relocations.get(event.robot_id, 0)
        if not pending:
            self.false_relocations += 1
            return
        self.recovered_external_discontinuities += 1
        self.external_recovery_duration_s += event.recovery_duration_s
        self._consume_expected(event.robot_id, pending)

    def handle_relocalization_failed(
        self,
        event: RelocalizationFailedEvent,
    ) -> None:
        pending = self._expected_relocations.get(event.robot_id, 0)
        if pending:
            self._consume_expected(event.robot_id, pending)

    def _consume_expected(self, robot_id: str, pending: int) -> None:
        if pending == 1:
            del self._expected_relocations[robot_id]
        else:
            self._expected_relocations[robot_id] = pending - 1

    @property
    def automatic_recovery_rate(self) -> float:
        if self.total_external_discontinuities == 0:
            return 0.0
        return min(
            1.0,
            self.recovered_external_discontinuities
            / self.total_external_discontinuities,
        )

    @property
    def mean_recovery_time_s(self) -> float:
        if self.recovered_external_discontinuities == 0:
            return 0.0
        return (
            self.external_recovery_duration_s / self.recovered_external_discontinuities
        )
