from dataclasses import dataclass

from entities.pose import Pose
from entities.value_validation import require_finite


@dataclass(frozen=True)
class AbsolutePoseFix:
    """A pose derived from validated absolute sensor evidence."""

    pose: Pose
    error_bound_m: float
    residual_m: float
    sim_time: float

    def __post_init__(self) -> None:
        if not isinstance(self.pose, Pose):
            raise TypeError("pose must be a Pose")
        require_finite("error_bound_m", self.error_bound_m, minimum=0.0)
        require_finite("residual_m", self.residual_m, minimum=0.0)
        require_finite("sim_time", self.sim_time, minimum=0.0)
