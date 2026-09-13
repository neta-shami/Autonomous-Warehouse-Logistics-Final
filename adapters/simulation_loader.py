"""Compile and validate the MuJoCo warehouse model."""

import re
from typing import Dict, Iterable, Tuple

import mujoco
import numpy as np

from adapters.mujoco_model_configuration_error import (
    MuJoCoModelConfigurationError,
)
from entities.enums import CardinalDirection, ZoneType
from entities.warehouse_topology import WarehouseTopology


class SimulationLoader:
    """Build physical and logical models and reject disagreement at startup."""

    _TOLERANCE = 1e-6
    _SHELF_GEOM_SUFFIXES = (
        "tier_0",
        "tier_1",
        "leg_sw",
        "leg_se",
        "leg_nw",
        "leg_ne",
    )
    _WALL_DIRECTIONS = (
        (CardinalDirection.POS_X, "wall_pos_x", (1.0, 0.0, 0.0)),
        (CardinalDirection.NEG_X, "wall_neg_x", (-1.0, 0.0, 0.0)),
        (CardinalDirection.POS_Y, "wall_pos_y", (0.0, 1.0, 0.0)),
        (CardinalDirection.NEG_Y, "wall_neg_y", (0.0, -1.0, 0.0)),
    )

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.model: mujoco.MjModel
        self.data: mujoco.MjData

    def load(self) -> Tuple[mujoco.MjModel, mujoco.MjData, WarehouseTopology]:
        """Compile the XML, construct topology, then validate their contract."""
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        topology = self._build_topology()
        self._validate_model(topology)
        return self.model, self.data, topology

    @staticmethod
    def _build_topology() -> WarehouseTopology:
        zone_map: Dict[Tuple[int, int], ZoneType] = {}
        for x in (2, 4, 6, 8):
            for y in range(2, 8):
                zone_map[(x, y)] = ZoneType.SHELF
        for x in range(10):
            zone_map[(x, 9)] = ZoneType.PARKING
        zone_map[(9, 1)] = ZoneType.INBOUND_DOCK
        zone_map[(9, 8)] = ZoneType.OUTBOUND_DOCK
        return WarehouseTopology(
            grid_width=10,
            grid_height=10,
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            zone_map=zone_map,
        )

    def _validate_model(self, topology: WarehouseTopology) -> None:
        self._validate_walls(topology)
        self._validate_shelves(topology)
        self._validate_zones(topology)
        self._validate_robot_sensors(topology)
        self._validate_robot_hardware()

    def _require_id(self, object_type, kind: str, name: str) -> int:
        element_id = mujoco.mj_name2id(self.model, object_type, name)
        if element_id == -1:
            raise MuJoCoModelConfigurationError(
                f"Required MuJoCo {kind} '{name}' is missing"
            )
        return element_id

    def _assert_close(self, label: str, actual, expected) -> None:
        if not np.allclose(actual, expected, atol=self._TOLERANCE, rtol=0.0):
            raise MuJoCoModelConfigurationError(
                f"{label} mismatch: compiled {actual}, expected {expected}"
            )

    def _validate_walls(self, topology: WarehouseTopology) -> None:
        wall_contract = {
            "wall_west": (0, topology.min_x, 1.0),
            "wall_east": (0, topology.max_x, -1.0),
            "wall_south": (1, topology.min_y, 1.0),
            "wall_north": (1, topology.max_y, -1.0),
        }
        for name, (axis, expected_bound, size_sign) in wall_contract.items():
            geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "geom", name)
            position = float(self.data.geom_xpos[geom_id, axis])
            half_size = float(self.model.geom_size[geom_id, axis])
            inner_surface = position + size_sign * half_size
            self._assert_close(name, inner_surface, expected_bound)

    def _validate_shelves(self, topology: WarehouseTopology) -> None:
        expected = {
            coords
            for coords, zone in topology.zone_map.items()
            if zone is ZoneType.SHELF
        }
        compiled = {}
        pattern = re.compile(r"^shelf_(-?\d+)_(-?\d+)$")
        for body_id in range(1, self.model.nbody):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            match = pattern.match(name or "")
            if match:
                compiled[(int(match.group(1)), int(match.group(2)))] = body_id
        if set(compiled) != expected:
            raise MuJoCoModelConfigurationError(
                "Compiled shelf coordinates do not match logical shelf zones: "
                f"compiled={sorted(compiled)}, expected={sorted(expected)}"
            )

        for (x, y), body_id in compiled.items():
            body_name = f"shelf_{x}_{y}"
            self._assert_close(
                body_name, self.data.xpos[body_id, :2], (float(x), float(y))
            )
            for suffix in self._SHELF_GEOM_SUFFIXES:
                geom_name = f"{body_name}_{suffix}"
                geom_id = self._require_id(
                    mujoco.mjtObj.mjOBJ_GEOM, "geom", geom_name
                )
                if int(self.model.geom_bodyid[geom_id]) != body_id:
                    raise MuJoCoModelConfigurationError(
                        f"Shelf geom '{geom_name}' is attached to the wrong body"
                    )

    def _validate_zones(self, topology: WarehouseTopology) -> None:
        zone_contract = {
            "zone_parking": ((4.5, 9.0), (5.0, 0.5)),
            "zone_inbound": ((9.0, 1.0), (0.5, 0.5)),
            "zone_outbound": ((9.0, 8.0), (0.5, 0.5)),
        }
        for name, (expected_position, expected_size) in zone_contract.items():
            geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "geom", name)
            self._assert_close(
                f"{name} position",
                self.data.geom_xpos[geom_id, :2],
                expected_position,
            )
            self._assert_close(
                f"{name} size", self.model.geom_size[geom_id, :2], expected_size
            )

        zone_expectations = {
            (9, 1): ZoneType.INBOUND_DOCK,
            (9, 8): ZoneType.OUTBOUND_DOCK,
        }
        for coords, expected_zone in zone_expectations.items():
            if topology.get_zone_type(*coords) is not expected_zone:
                raise MuJoCoModelConfigurationError(
                    f"Logical zone at {coords} does not match {expected_zone.name}"
                )

    def _robot_ids(self) -> Iterable[str]:
        pattern = re.compile(r"^robot_([^_]+)$")
        robot_ids = []
        for body_id in range(1, self.model.nbody):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            match = pattern.match(name or "")
            if match:
                robot_ids.append(match.group(1))
        if not robot_ids:
            raise MuJoCoModelConfigurationError(
                "The compiled model contains no root robot bodies"
            )
        return tuple(robot_ids)

    def _validate_robot_sensors(self, topology: WarehouseTopology) -> None:
        obstacle_top = self._highest_stationary_obstacle(topology)
        wall_top = min(
            self._geom_top(name)
            for name in ("wall_west", "wall_east", "wall_south", "wall_north")
        )
        for robot_id in self._robot_ids():
            body_id = self._require_id(
                mujoco.mjtObj.mjOBJ_BODY, "body", f"robot_{robot_id}"
            )
            for _, suffix, expected_direction in self._WALL_DIRECTIONS:
                site_name = f"robot_{robot_id}_{suffix}_site"
                sensor_name = f"robot_{robot_id}_{suffix}"
                site_id = self._require_id(
                    mujoco.mjtObj.mjOBJ_SITE, "site", site_name
                )
                self._validate_scalar_sensor(
                    sensor_name, mujoco.mjtSensor.mjSENS_RANGEFINDER, site_id
                )
                if int(self.model.site_bodyid[site_id]) != body_id:
                    raise MuJoCoModelConfigurationError(
                        f"Wall sensor site '{site_name}' is not a chassis child"
                    )
                self._assert_close(
                    f"{site_name} X/Y offset",
                    self.data.site_xpos[site_id, :2],
                    self.data.xpos[body_id, :2],
                )
                site_direction = self.data.site_xmat[site_id].reshape(3, 3)[:, 2]
                self._assert_close(
                    f"{site_name} direction", site_direction, expected_direction
                )
                height = float(self.data.site_xpos[site_id, 2])
                if not obstacle_top < height < wall_top:
                    raise MuJoCoModelConfigurationError(
                        f"{site_name} world height {height:.3f} must be above "
                        f"obstacles ({obstacle_top:.3f}) and below walls "
                        f"({wall_top:.3f})"
                    )

            for axis in ("x", "y"):
                joint_id = self._require_id(
                    mujoco.mjtObj.mjOBJ_JOINT,
                    "joint",
                    f"robot_{robot_id}_{axis}",
                )
                self._validate_scalar_sensor(
                    f"robot_{robot_id}_velocity_{axis}",
                    mujoco.mjtSensor.mjSENS_JOINTVEL,
                    joint_id,
                )

            payload_site = self._require_id(
                mujoco.mjtObj.mjOBJ_SITE,
                "site",
                f"robot_{robot_id}_payload_site",
            )
            self._validate_scalar_sensor(
                f"robot_{robot_id}_payload",
                mujoco.mjtSensor.mjSENS_RANGEFINDER,
                payload_site,
            )
            payload_direction = self.data.site_xmat[payload_site].reshape(3, 3)[:, 2]
            self._assert_close(
                f"robot_{robot_id}_payload_site direction",
                payload_direction,
                (0.0, 0.0, 1.0),
            )
            gripper_id = self._require_id(
                mujoco.mjtObj.mjOBJ_BODY,
                "body",
                f"robot_{robot_id}_gripper",
            )
            if int(self.model.site_bodyid[payload_site]) != gripper_id:
                raise MuJoCoModelConfigurationError(
                    f"robot_{robot_id}_payload_site is not on the moving gripper"
                )

    def _validate_scalar_sensor(
        self, name: str, expected_type, expected_object_id: int
    ) -> None:
        sensor_id = self._require_id(mujoco.mjtObj.mjOBJ_SENSOR, "sensor", name)
        if self.model.sensor_type[sensor_id] != expected_type:
            raise MuJoCoModelConfigurationError(
                f"Sensor '{name}' has the wrong MuJoCo sensor type"
            )
        if int(self.model.sensor_objid[sensor_id]) != expected_object_id:
            raise MuJoCoModelConfigurationError(
                f"Sensor '{name}' targets the wrong model element"
            )
        if int(self.model.sensor_dim[sensor_id]) != 1:
            raise MuJoCoModelConfigurationError(
                f"Sensor '{name}' must produce exactly one value"
            )

    def _validate_robot_hardware(self) -> None:
        """Reject code/model mismatches for every commanded robot element."""
        for robot_id in self._robot_ids():
            gripper_id = self._require_id(
                mujoco.mjtObj.mjOBJ_BODY,
                "body",
                f"robot_{robot_id}_gripper",
            )
            grip_site_name = f"robot_{robot_id}_grip_site"
            grip_site_id = self._require_id(
                mujoco.mjtObj.mjOBJ_SITE, "site", grip_site_name
            )
            if int(self.model.site_bodyid[grip_site_id]) != gripper_id:
                raise MuJoCoModelConfigurationError(
                    f"Grip site '{grip_site_name}' is not on the moving gripper"
                )

            for suffix in ("x", "y", "lift", "turret", "extend"):
                joint_name = f"robot_{robot_id}_{suffix}"
                joint_id = self._require_id(
                    mujoco.mjtObj.mjOBJ_JOINT, "joint", joint_name
                )
                actuator_id = self._require_id(
                    mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator", joint_name
                )
                if int(self.model.actuator_trnid[actuator_id, 0]) != joint_id:
                    raise MuJoCoModelConfigurationError(
                        f"Actuator '{joint_name}' targets the wrong joint"
                    )

    def _geom_top(self, name: str) -> float:
        geom_id = self._require_id(mujoco.mjtObj.mjOBJ_GEOM, "geom", name)
        return float(
            self.data.geom_xpos[geom_id, 2] + self.model.geom_size[geom_id, 2]
        )

    def _highest_stationary_obstacle(self, topology: WarehouseTopology) -> float:
        highest = 0.0
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            ) or ""
            x, y, z = (float(value) for value in self.data.geom_xpos[geom_id])
            inside = (
                topology.min_x <= x <= topology.max_x
                and topology.min_y <= y <= topology.max_y
            )
            if inside and (
                body_name.startswith("shelf_")
                or body_name.startswith("package_")
            ):
                highest = max(highest, z + float(self.model.geom_size[geom_id, 2]))
        return highest
