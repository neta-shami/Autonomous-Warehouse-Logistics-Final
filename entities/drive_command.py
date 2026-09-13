from dataclasses import dataclass

from entities.value_validation import require_finite


@dataclass(frozen=True)
class DriveCommand:
    """A validated holonomic velocity command in metres per second."""

    vx: float
    vy: float

    def __post_init__(self) -> None:
        require_finite("drive vx", self.vx)
        require_finite("drive vy", self.vy)
