from dataclasses import dataclass
from math import hypot

from entities.value_validation import require_finite


@dataclass(frozen=True)
class Pose:
    """Immutable believed or measured point on the warehouse floor."""

    x: float
    y: float

    def __post_init__(self) -> None:
        require_finite("pose x", self.x)
        require_finite("pose y", self.y)

    def distance_to(self, other: "Pose") -> float:
        if not isinstance(other, Pose):
            raise TypeError("other must be a Pose")
        return hypot(self.x - other.x, self.y - other.y)
