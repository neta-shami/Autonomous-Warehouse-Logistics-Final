from abc import ABC, abstractmethod


class ISimulationClock(ABC):
    """Port exposing monotonic simulation time to deadline-based use cases."""

    @abstractmethod
    def get_time(self) -> float:
        """Return current simulation time in seconds."""
        pass
