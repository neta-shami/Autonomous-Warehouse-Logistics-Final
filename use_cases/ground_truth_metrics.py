"""Metrics-only physical motion and localization measurements."""

from typing import Dict, FrozenSet, Iterable, Optional

from entities.enums import FaultCode, LocalizationStatus, PoseSource
from entities.pose import Pose
from entities.robot_state import RobotState
from entities.value_validation import require_finite
from interfaces.i_ground_truth_probe import IGroundTruthProbe
from interfaces.sensor_read_error import SensorReadError


class GroundTruthMetrics:
    """Own all measurements that require the isolated ground-truth port."""

    def __init__(self, truth_probe: Optional[IGroundTruthProbe]) -> None:
        self.truth_probe = truth_probe
        self.total_distance_driven = 0.0
        self.localization_errors: list[float] = []
        self.localization_grid_samples = 0
        self.localization_grid_matches = 0
        self.healthy_localization_samples = 0
        self.absolute_fix_samples = 0
        self.initialization_attempts: Dict[str, float] = {}
        self.initialization_durations_s: Dict[str, float] = {}
        self._last_truth_pose: Dict[str, Pose] = {}

    def reset_motion_baseline(self, robot_id: str) -> None:
        self._last_truth_pose.pop(robot_id, None)

    def sample_motion(
        self,
        robot_ids: Iterable[str],
        sim_time: float,
    ) -> Dict[str, float]:
        """Return each robot's displacement since its previous truth sample."""
        require_finite("motion sample sim_time", sim_time, minimum=0.0)
        displacement_by_robot: Dict[str, float] = {}
        if self.truth_probe is None:
            return displacement_by_robot
        for robot_id in robot_ids:
            current = self._read_pose(robot_id)
            if current is None:
                self.reset_motion_baseline(robot_id)
                continue
            previous = self._last_truth_pose.get(robot_id)
            if previous is not None:
                displacement = previous.distance_to(current)
                self.total_distance_driven += displacement
                displacement_by_robot[robot_id] = displacement
            self._last_truth_pose[robot_id] = current
        return displacement_by_robot

    def sample_localization(
        self,
        robots: Iterable[RobotState],
        excluded_robot_ids: FrozenSet[str],
        sim_time: Optional[float] = None,
    ) -> None:
        """Measure belief error without feeding truth back into control."""
        for robot in robots:
            sample_time = self._sample_time(robot, sim_time)
            self._record_initialization(robot, sample_time)
            self._record_fix_availability(robot)
            if robot.robot_id in excluded_robot_ids or robot.pose_estimate is None:
                continue
            truth = self._read_pose(robot.robot_id)
            if truth is None:
                continue
            estimate = robot.pose_estimate.pose
            self.localization_errors.append(estimate.distance_to(truth))
            self.localization_grid_samples += 1
            if (round(estimate.x), round(estimate.y)) == (
                round(truth.x),
                round(truth.y),
            ):
                self.localization_grid_matches += 1

    def _read_pose(self, robot_id: str) -> Optional[Pose]:
        if self.truth_probe is None:
            return None
        try:
            return self.truth_probe.true_pose(robot_id)
        except SensorReadError:
            return None

    @staticmethod
    def _sample_time(robot: RobotState, sim_time: Optional[float]) -> float:
        sample_time = (
            sim_time
            if sim_time is not None
            else (
                robot.pose_estimate.sim_time if robot.pose_estimate is not None else 0.0
            )
        )
        require_finite("localization sample sim_time", sample_time, minimum=0.0)
        return sample_time

    def _record_initialization(self, robot: RobotState, sample_time: float) -> None:
        self.initialization_attempts.setdefault(robot.robot_id, sample_time)
        if (
            robot.robot_id not in self.initialization_durations_s
            and robot.localization_status is LocalizationStatus.TRUSTED
            and robot.pose_estimate is not None
        ):
            self.initialization_durations_s[robot.robot_id] = (
                sample_time - self.initialization_attempts[robot.robot_id]
            )

    def _record_fix_availability(self, robot: RobotState) -> None:
        if FaultCode.SENSOR_INVALID in robot.active_faults:
            return
        self.healthy_localization_samples += 1
        if (
            robot.pose_estimate is not None
            and robot.pose_estimate.source is not PoseSource.PREDICTED
        ):
            self.absolute_fix_samples += 1

    @property
    def mean_localization_error_m(self) -> float:
        if not self.localization_errors:
            return 0.0
        return sum(self.localization_errors) / len(self.localization_errors)

    @property
    def peak_localization_error_m(self) -> float:
        return max(self.localization_errors, default=0.0)

    @property
    def grid_cell_accuracy(self) -> float:
        if self.localization_grid_samples == 0:
            return 0.0
        return self.localization_grid_matches / self.localization_grid_samples

    @property
    def absolute_fix_availability(self) -> float:
        if self.healthy_localization_samples == 0:
            return 0.0
        return self.absolute_fix_samples / self.healthy_localization_samples

    @property
    def initialization_success_rate(self) -> float:
        if not self.initialization_attempts:
            return 0.0
        return len(self.initialization_durations_s) / len(self.initialization_attempts)
