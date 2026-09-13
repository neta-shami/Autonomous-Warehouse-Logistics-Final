"""Deterministic test adapter for physical and sensor fault injection."""

import math
import random
from typing import Callable, Dict, Tuple

from entities.enums import CardinalDirection
from entities.pose import Pose
from entities.sensor_frame import SensorFrame
from interfaces.i_sensor_adapter import ISensorAdapter
from interfaces.sensor_read_error import SensorReadError
from use_cases.metrics_collector import MetricsCollector


class FaultInjector(ISensorAdapter):
    """Decorate a sensor port and coordinate explicitly seeded fault actions."""

    def __init__(
        self,
        sensor_adapter: ISensorAdapter,
        metrics: MetricsCollector,
        *,
        random_seed: int,
    ) -> None:
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise TypeError("random_seed must be an integer")
        self._sensor_adapter = sensor_adapter
        self._metrics = metrics
        self._rng = random.Random(random_seed)
        self._dropouts: Dict[str, int] = {}
        self._ray_overrides: Dict[
            Tuple[str, CardinalDirection], float
        ] = {}
        self._ray_biases: Dict[Tuple[str, CardinalDirection], float] = {}
        self._gradual_biases: Dict[
            Tuple[str, CardinalDirection], Tuple[float, float]
        ] = {}

    def read_frame(self, robot_id: str) -> SensorFrame:
        remaining = self._dropouts.get(robot_id, 0)
        if remaining:
            self._dropouts[robot_id] = remaining - 1
            raise SensorReadError(
                f"Injected sensor dropout for robot '{robot_id}'"
            )
        frame = self._sensor_adapter.read_frame(robot_id)
        readings = []
        for direction, original in frame.wall_ranges:
            key = (robot_id, direction)
            value = self._ray_overrides.get(key, original)
            if key in self._gradual_biases and value != -1.0:
                current, step = self._gradual_biases[key]
                current += step
                self._gradual_biases[key] = (current, step)
                value += current
            if key in self._ray_biases and value != -1.0:
                value += self._ray_biases[key]
            readings.append((direction, max(-1.0, value)))
        return SensorFrame(
            frame.robot_id,
            frame.observed_chassis_velocity,
            tuple(readings),
            frame.payload_range,
            frame.sim_time,
            frame.sequence,
        )

    def drop_next_frames(self, robot_id: str, count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("dropout count must be a positive integer")
        self._dropouts[robot_id] = self._dropouts.get(robot_id, 0) + count

    def occlude_ray(
        self, robot_id: str, direction: CardinalDirection
    ) -> None:
        self._require_direction(direction)
        self._ray_overrides[(robot_id, direction)] = -1.0

    def override_ray(
        self,
        robot_id: str,
        direction: CardinalDirection,
        distance_m: float,
    ) -> None:
        self._require_direction(direction)
        if (
            isinstance(distance_m, bool)
            or not math.isfinite(distance_m)
            or distance_m < -1.0
        ):
            raise ValueError("injected range must be finite and at least -1")
        self._ray_overrides[(robot_id, direction)] = distance_m

    def inject_seeded_single_ray_bias(
        self, robot_id: str, maximum_absolute_bias_m: float
    ) -> CardinalDirection:
        if (
            isinstance(maximum_absolute_bias_m, bool)
            or not math.isfinite(maximum_absolute_bias_m)
            or maximum_absolute_bias_m <= 0.0
        ):
            raise ValueError("maximum bias must be finite and positive")
        direction = self._rng.choice(tuple(CardinalDirection))
        magnitude = self._rng.uniform(
            maximum_absolute_bias_m / 2.0, maximum_absolute_bias_m
        )
        sign = self._rng.choice((-1.0, 1.0))
        self._ray_biases[(robot_id, direction)] = sign * magnitude
        return direction

    def set_gradual_ray_bias(
        self,
        robot_id: str,
        direction: CardinalDirection,
        *,
        step_m: float,
    ) -> None:
        self._require_direction(direction)
        if isinstance(step_m, bool) or not math.isfinite(step_m) or step_m == 0.0:
            raise ValueError("bias step must be a finite non-zero number")
        self._gradual_biases[(robot_id, direction)] = (0.0, step_m)

    def clear_sensor_faults(self, robot_id: str) -> None:
        self._dropouts.pop(robot_id, None)
        for mapping in (
            self._ray_overrides,
            self._ray_biases,
            self._gradual_biases,
        ):
            for key in tuple(mapping):
                if key[0] == robot_id:
                    mapping.pop(key)

    def teleport_robot(
        self,
        robot_id: str,
        pose: Pose,
        sim_time: float,
        teleport: Callable[[str, Pose], None],
    ) -> None:
        """Reset the distance baseline immediately before moving ground truth."""
        if not isinstance(pose, Pose):
            raise TypeError("pose must be Pose")
        if not callable(teleport):
            raise TypeError("teleport must be callable")
        self._metrics.mark_external_discontinuity(robot_id, sim_time)
        teleport(robot_id, pose)

    @staticmethod
    def inject_drive_stall(
        robot_id: str,
        stalled: bool,
        set_stalled: Callable[[str, bool], None],
    ) -> None:
        if not isinstance(stalled, bool) or not callable(set_stalled):
            raise TypeError("stalled must be bool and set_stalled must be callable")
        set_stalled(robot_id, stalled)

    @staticmethod
    def steal_payload(
        payload_id: str,
        detach_payload: Callable[[str], object],
        remove_payload: Callable[[str], None],
    ) -> None:
        if not isinstance(payload_id, str) or not payload_id:
            raise ValueError("payload_id must be a non-empty string")
        if not callable(detach_payload):
            raise TypeError("detach_payload must be callable")
        if not callable(remove_payload):
            raise TypeError("remove_payload must be callable")
        detach_payload(payload_id)
        remove_payload(payload_id)

    @staticmethod
    def _require_direction(direction: CardinalDirection) -> None:
        if not isinstance(direction, CardinalDirection):
            raise TypeError("direction must be CardinalDirection")
