"""MuJoCo adapter for atomic robot-mounted sensor frames."""

import math
import random
from typing import Dict

import mujoco

from entities.enums import CardinalDirection
from entities.sensor_frame import SensorFrame
from interfaces.i_sensor_adapter import ISensorAdapter
from interfaces.sensor_read_error import SensorReadError


class MuJoCoSensorAdapter(ISensorAdapter):
    """Read named MuJoCo sensors without deriving pose or payload state."""

    _WALL_SENSORS = (
        (CardinalDirection.POS_X, "wall_pos_x"),
        (CardinalDirection.NEG_X, "wall_neg_x"),
        (CardinalDirection.POS_Y, "wall_pos_y"),
        (CardinalDirection.NEG_Y, "wall_neg_y"),
    )

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        random_seed: int = 0,
        range_noise_bound_m: float = 0.005,
        velocity_noise_bound_mps: float = 0.0,
    ):
        if range_noise_bound_m < 0.0 or not math.isfinite(range_noise_bound_m):
            raise ValueError("range_noise_bound_m must be finite and non-negative")
        if (
            velocity_noise_bound_mps < 0.0
            or not math.isfinite(velocity_noise_bound_mps)
        ):
            raise ValueError(
                "velocity_noise_bound_mps must be finite and non-negative"
            )
        self.model = model
        self.data = data
        self.range_noise_bound_m = range_noise_bound_m
        self.velocity_noise_bound_mps = velocity_noise_bound_mps
        self._random_seed = random_seed
        self._rng_by_robot: Dict[str, random.Random] = {}
        self._sensor_addresses: Dict[str, Dict[str, int]] = {}
        self._last_sim_time: Dict[str, float] = {}
        self._next_sequence: Dict[str, int] = {}

    def read_frame(self, robot_id: str) -> SensorFrame:
        """Sample one coherent frame, rejecting stale or cross-wired input."""
        if not isinstance(robot_id, str) or not robot_id:
            raise SensorReadError("robot_id must be a non-empty string")
        addresses = self._addresses_for(robot_id)
        sim_time = float(self.data.time)
        if not math.isfinite(sim_time) or sim_time < 0.0:
            raise SensorReadError(
                f"Robot '{robot_id}' sensor timestamp is invalid: {sim_time}"
            )
        previous_time = self._last_sim_time.get(robot_id)
        if previous_time is not None and sim_time <= previous_time:
            raise SensorReadError(
                f"Robot '{robot_id}' sensor timestamp did not advance: "
                f"{sim_time} <= {previous_time}"
            )

        wall_ranges = tuple(
            (
                direction,
                self._noisy_range(
                    robot_id,
                    self._read_range(robot_id, addresses[suffix], suffix),
                ),
            )
            for direction, suffix in self._WALL_SENSORS
        )
        velocity = (
            self._noisy_velocity(
                robot_id,
                self._read_finite(robot_id, addresses["velocity_x"], "velocity_x"),
            ),
            self._noisy_velocity(
                robot_id,
                self._read_finite(robot_id, addresses["velocity_y"], "velocity_y"),
            ),
        )
        payload_range = self._noisy_range(
            robot_id,
            self._read_range(robot_id, addresses["payload"], "payload"),
        )
        sequence = self._next_sequence.get(robot_id, 0)
        frame = SensorFrame(
            robot_id=robot_id,
            observed_chassis_velocity=velocity,
            wall_ranges=wall_ranges,
            payload_range=payload_range,
            sim_time=sim_time,
            sequence=sequence,
        )
        # History advances only after the complete immutable frame validates.
        self._last_sim_time[robot_id] = sim_time
        self._next_sequence[robot_id] = sequence + 1
        return frame

    def _addresses_for(self, robot_id: str) -> Dict[str, int]:
        cached = self._sensor_addresses.get(robot_id)
        if cached is not None:
            return cached

        body_name = f"robot_{robot_id}"
        if mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        ) == -1:
            raise SensorReadError(
                f"Cannot read unknown robot '{robot_id}': body '{body_name}' is missing"
            )

        suffixes = [suffix for _, suffix in self._WALL_SENSORS]
        suffixes.extend(("velocity_x", "velocity_y", "payload"))
        addresses: Dict[str, int] = {}
        for suffix in suffixes:
            sensor_name = f"robot_{robot_id}_{suffix}"
            sensor_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
            )
            if sensor_id == -1:
                raise SensorReadError(
                    f"Robot '{robot_id}' sensor '{sensor_name}' is missing"
                )
            if int(self.model.sensor_dim[sensor_id]) != 1:
                raise SensorReadError(
                    f"Robot '{robot_id}' sensor '{sensor_name}' must be scalar"
                )
            addresses[suffix] = int(self.model.sensor_adr[sensor_id])
        self._sensor_addresses[robot_id] = addresses
        return addresses

    def _read_finite(self, robot_id: str, address: int, label: str) -> float:
        try:
            value = float(self.data.sensordata[address])
        except (IndexError, TypeError, ValueError) as exc:
            raise SensorReadError(
                f"Robot '{robot_id}' could not read sensor '{label}'"
            ) from exc
        if not math.isfinite(value):
            raise SensorReadError(
                f"Robot '{robot_id}' sensor '{label}' is non-finite"
            )
        return value

    def _read_range(self, robot_id: str, address: int, label: str) -> float:
        value = self._read_finite(robot_id, address, label)
        if value < 0.0 and value != -1.0:
            raise SensorReadError(
                f"Robot '{robot_id}' sensor '{label}' returned invalid range {value}"
            )
        return value

    def _rng_for(self, robot_id: str) -> random.Random:
        rng = self._rng_by_robot.get(robot_id)
        if rng is None:
            stable_id = sum(
                (index + 1) * byte
                for index, byte in enumerate(robot_id.encode("utf-8"))
            )
            rng = random.Random(self._random_seed + stable_id)
            self._rng_by_robot[robot_id] = rng
        return rng

    def _noisy_range(self, robot_id: str, value: float) -> float:
        # -1 is valid MuJoCo "no hit" evidence, not a value to perturb.
        if value == -1.0 or self.range_noise_bound_m == 0.0:
            return value
        noise = self._rng_for(robot_id).uniform(
            -self.range_noise_bound_m, self.range_noise_bound_m
        )
        return max(0.0, value + noise)

    def _noisy_velocity(self, robot_id: str, value: float) -> float:
        if self.velocity_noise_bound_mps == 0.0:
            return value
        return value + self._rng_for(robot_id).uniform(
            -self.velocity_noise_bound_mps, self.velocity_noise_bound_mps
        )
