from dataclasses import dataclass

from entities.enums import PoseSource
from entities.pose import Pose
from entities.value_validation import require_finite


@dataclass(frozen=True)
class PoseEstimate:
    """A believed pose with its conservative error bound and provenance."""

    pose: Pose
    uncertainty_m: float
    source: PoseSource
    sim_time: float

    def __post_init__(self) -> None:
        if not isinstance(self.pose, Pose):
            raise TypeError("pose must be a Pose")
        if not isinstance(self.source, PoseSource):
            raise TypeError("source must be a PoseSource")
        require_finite("uncertainty_m", self.uncertainty_m, minimum=0.0)
        require_finite("sim_time", self.sim_time, minimum=0.0)
