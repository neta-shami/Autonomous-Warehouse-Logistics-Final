"""Simulation-time upper bounds for ordinary FSM states."""

from typing import Mapping, Optional

from entities.enums import FaultCode, FaultSeverity, FSMStatus
from entities.health import HealthCheckContext, HealthFinding
from entities.value_validation import require_finite
from interfaces.i_health_check import IHealthCheck


class StateTimeoutCheck(IHealthCheck):
    """Report states that exceed configured simulation-time bounds."""

    DEFAULT_TIMEOUTS = {
        FSMStatus.NAVIGATING: 120.0,
        FSMStatus.AVOIDING: 10.0,
        FSMStatus.MANIPULATING: 5.0,
    }

    def __init__(
        self, timeouts: Optional[Mapping[FSMStatus, float]] = None
    ) -> None:
        configured = self.DEFAULT_TIMEOUTS if timeouts is None else timeouts
        self.timeouts = {}
        for status, timeout in configured.items():
            if not isinstance(status, FSMStatus):
                raise TypeError("timeout keys must be FSMStatus")
            require_finite("state timeout", timeout, minimum=0.0)
            if timeout == 0.0:
                raise ValueError("state timeout must be greater than zero")
            self.timeouts[status] = timeout

    def evaluate(
        self, context: HealthCheckContext
    ) -> Optional[HealthFinding]:
        timeout = self.timeouts.get(context.fsm_status)
        if timeout is None or context.state_duration_s < timeout:
            return None
        return HealthFinding(
            FaultCode.STATE_TIMEOUT, FaultSeverity.STOP, True, True, None
        )
