from abc import ABC, abstractmethod
from typing import Optional

from entities.health import HealthCheckContext, HealthFinding


class IHealthCheck(ABC):
    """One independently replaceable robot-health rule."""

    @abstractmethod
    def evaluate(self, context: HealthCheckContext) -> Optional[HealthFinding]:
        """Return one finding or None when the check is healthy."""
        raise NotImplementedError
