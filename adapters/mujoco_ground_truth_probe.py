"""Metrics-only access to MuJoCo ground truth."""

import math

import mujoco

from entities.pose import Pose
from interfaces.i_ground_truth_probe import IGroundTruthProbe
from interfaces.sensor_read_error import SensorReadError


class MuJoCoGroundTruthProbe(IGroundTruthProbe):
    """Read physical robot positions for KPI measurement, never control."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._model = model
        self._data = data

    def true_pose(self, robot_id: str) -> Pose:
        """Return the named robot body's finite physical X/Y position."""
        body_name = f"robot_{robot_id}"
        body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id == -1:
            raise SensorReadError(
                f"Cannot measure unknown robot '{robot_id}': "
                f"MuJoCo body '{body_name}' is missing"
            )
        x, y = (float(value) for value in self._data.xpos[body_id, :2])
        if not math.isfinite(x) or not math.isfinite(y):
            raise SensorReadError(
                f"Robot '{robot_id}' has a non-finite ground-truth pose"
            )
        return Pose(x=x, y=y)
