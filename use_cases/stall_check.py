"""Persistent commanded-motion versus observed-motion health check."""

from math import hypot
from typing import Dict, Optional

from entities.enums import FaultCode, FaultSeverity, FSMStatus
from entities.health import HealthCheckContext, HealthFinding
from entities.value_validation import require_finite
from interfaces.i_health_check import IHealthCheck


class StallCheck(IHealthCheck):
    """Detect a drive fault without confusing deliberate stops for stalls."""

    def __init__(
        self,
        persistence_s: float = 0.5,
        command_threshold: float = 0.05,
        velocity_threshold: float = 0.02,
    ) -> None:
        require_finite("persistence_s", persistence_s, minimum=0.0)
        require_finite("command_threshold", command_threshold, minimum=0.0)
        require_finite("velocity_threshold", velocity_threshold, minimum=0.0)
        self.persistence_s = persistence_s
        self.command_threshold = command_threshold
        self.velocity_threshold = velocity_threshold
        self._since: Dict[str, float] = {}

    def evaluate(
        self, context: HealthCheckContext
    ) -> Optional[HealthFinding]:
        robot_id = context.robot.robot_id
        velocity = context.robot.observed_chassis_velocity
        active = (
            context.fsm_status is FSMStatus.NAVIGATING
            and hypot(
                context.last_applied_drive_command.vx,
                context.last_applied_drive_command.vy,
            )
            >= self.command_threshold
            and velocity is not None
            and hypot(*velocity) <= self.velocity_threshold
        )
        if not active:
            self._since.pop(robot_id, None)
            return None
        since = self._since.setdefault(robot_id, context.sim_time)
        duration = context.sim_time - since
        if duration < self.persistence_s:
            return None
        return HealthFinding(
            FaultCode.DRIVE_STALL,
            FaultSeverity.STOP,
            True,
            True,
            None,
        )
