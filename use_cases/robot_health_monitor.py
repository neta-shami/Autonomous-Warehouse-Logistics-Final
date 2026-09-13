"""Composition root for independent robot health checks."""

from typing import Iterable, Tuple

from entities.enums import FaultCode, LocalizationStatus
from entities.health import HealthCheckContext, HealthFinding
from interfaces.i_health_check import IHealthCheck


class RobotHealthMonitor:
    """Run configured checks without embedding any health rule itself."""

    def __init__(self, checks: Iterable[IHealthCheck]):
        self._checks = tuple(checks)

    def evaluate(
        self, context: HealthCheckContext
    ) -> Tuple[HealthFinding, ...]:
        """Return all simultaneous findings in registration order."""
        return tuple(
            finding
            for check in self._checks
            for finding in (check.evaluate(context),)
            if finding is not None
        )

    def can_clear(
        self, fault: FaultCode, context: HealthCheckContext
    ) -> bool:
        """Return whether explicit fault-specific reset evidence is sufficient."""
        if fault is FaultCode.SENSOR_INVALID:
            return (
                context.frame is not None
                and context.robot.localization_status is LocalizationStatus.TRUSTED
            )
        if fault is FaultCode.DRIVE_STALL:
            return (
                context.drive_diagnostic_passed
                and context.drive_cause_removed
            )
        if fault in (
            FaultCode.PAYLOAD_MISMATCH,
            FaultCode.MANIPULATOR_FAILURE,
        ):
            return context.inventory_reconciled
        if fault is FaultCode.STATE_TIMEOUT:
            return context.state_duration_s == 0.0
        if fault is FaultCode.POSITION_PHYSICALLY_INVALID:
            return (
                context.position_feasible
                and context.robot.pose_estimate is not None
                and context.robot.localization_status
                in (LocalizationStatus.TRUSTED, LocalizationStatus.DEGRADED)
            )
        return False
