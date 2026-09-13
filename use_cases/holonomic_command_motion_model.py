from entities.drive_command import DriveCommand
from entities.pose import Pose
from entities.value_validation import require_finite
from interfaces.i_motion_prediction_model import IMotionPredictionModel


class HolonomicCommandMotionModel(IMotionPredictionModel):
    """Integrate an applied X/Y velocity command over a short interval."""

    def predict(
        self, previous: Pose, command: DriveCommand, dt: float
    ) -> Pose:
        if not isinstance(previous, Pose):
            raise TypeError("previous must be a Pose")
        if not isinstance(command, DriveCommand):
            raise TypeError("command must be a DriveCommand")
        require_finite("prediction dt", dt, minimum=0.0)
        return Pose(
            previous.x + command.vx * dt,
            previous.y + command.vy * dt,
        )
