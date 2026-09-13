from dataclasses import dataclass
from typing import Optional, Tuple

from entities.enums import CardinalDirection
from entities.value_validation import (
    require_finite,
    require_nonnegative_integer,
    validate_optional_velocity,
)


@dataclass(frozen=True)
class SensorFrame:
    """Immutable raw readings sampled for one robot at one simulation instant."""

    robot_id: str
    observed_chassis_velocity: Optional[Tuple[float, float]]
    wall_ranges: Tuple[Tuple[CardinalDirection, float], ...]
    payload_range: Optional[float]
    sim_time: float
    sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.robot_id, str) or not self.robot_id:
            raise ValueError("robot_id must be a non-empty string")
        validate_optional_velocity(self.observed_chassis_velocity)
        if not isinstance(self.wall_ranges, tuple):
            raise TypeError("wall_ranges must be an immutable tuple")

        seen = set()
        for reading in self.wall_ranges:
            if not isinstance(reading, tuple) or len(reading) != 2:
                raise TypeError("each wall range must be a direction/value tuple")
            direction, distance = reading
            if not isinstance(direction, CardinalDirection):
                raise TypeError("wall range direction must be CardinalDirection")
            if direction in seen:
                raise ValueError(f"Duplicate wall range direction: {direction.name}")
            seen.add(direction)
            require_finite("wall range", distance, minimum=-1.0)

        if self.payload_range is not None:
            require_finite("payload range", self.payload_range, minimum=-1.0)
        require_finite("sim_time", self.sim_time, minimum=0.0)
        require_nonnegative_integer("sequence", self.sequence)

    def range_for(self, direction: CardinalDirection) -> Optional[float]:
        """Return a valid hit, or None for a missing/no-hit direction."""
        if not isinstance(direction, CardinalDirection):
            raise TypeError("direction must be CardinalDirection")
        for reading_direction, distance in self.wall_ranges:
            if reading_direction is direction:
                return None if distance == -1.0 else distance
        return None
