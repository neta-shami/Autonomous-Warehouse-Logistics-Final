"""Phase-aware stable payload-presence health check."""

from typing import Dict, Optional

from entities.enums import FaultCode, FaultSeverity, TaskExecutionPhase
from entities.events import PayloadMismatchEvent
from entities.health import HealthCheckContext, HealthFinding
from interfaces.i_health_check import IHealthCheck


class PayloadCheck(IHealthCheck):
    """Compare debounced same-frame presence with the active transfer phase."""

    def evaluate(
        self, context: HealthCheckContext
    ) -> Optional[HealthFinding]:
        phase = context.task_phase
        if context.robot.expected_payload_id is None:
            return None
        if phase in (TaskExecutionPhase.PICKING, TaskExecutionPhase.DROPPING):
            return None
        expected_by_phase: Dict[TaskExecutionPhase, bool] = {
            TaskExecutionPhase.TO_SOURCE: False,
            TaskExecutionPhase.TO_TARGET: True,
        }
        expected = None if phase is None else expected_by_phase.get(phase)
        observed = context.payload_present
        if expected is None or observed is None or observed is expected:
            return None
        return HealthFinding(
            FaultCode.PAYLOAD_MISMATCH,
            FaultSeverity.QUARANTINE,
            True,
            True,
            PayloadMismatchEvent(
                context.sim_time,
                context.robot.robot_id,
            ),
        )
