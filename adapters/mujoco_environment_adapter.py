from typing import Iterable, Optional, Set, Tuple

import mujoco

from adapters.mujoco_model_configuration_error import (
    MuJoCoModelConfigurationError,
)
from entities.pose import Pose
from entities.value_validation import require_finite
from interfaces.i_environment import IEnvironment


class MuJoCoEnvironmentAdapter(IEnvironment):
    """
    Adapter bridging the high-level logic to the low-level MuJoCo physics engine.
    """
    PACKAGE_POOL_MIN_COORDINATE_M = 10.5

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._inactive_payload_ids = self._discover_inactive_payloads()

    def _discover_inactive_payloads(self) -> Set[str]:
        """Read named package bodies initially parked outside the warehouse."""
        inactive: Set[str] = set()
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            if not joint_name or not joint_name.startswith("package_"):
                continue
            qpos = int(self.model.jnt_qposadr[joint_id])
            if (
                self.data.qpos[qpos] >= self.PACKAGE_POOL_MIN_COORDINATE_M
                or self.data.qpos[qpos + 1]
                >= self.PACKAGE_POOL_MIN_COORDINATE_M
            ):
                inactive.add(joint_name.removeprefix("package_").removesuffix("_free"))
        return inactive

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def get_time(self) -> float:
        return float(self.data.time)

    def teleport_robot(self, robot_id: str, pose: Pose) -> None:
        """Fault-injection hook that moves named chassis slide joints."""
        if not isinstance(robot_id, str) or not robot_id:
            raise ValueError("robot_id must be a non-empty string")
        if not isinstance(pose, Pose):
            raise TypeError("pose must be Pose")
        body_name = f"robot_{robot_id}"
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id == -1:
            raise MuJoCoModelConfigurationError(
                f"Robot '{robot_id}' requires body '{body_name}' for teleport"
            )
        current_world = tuple(float(value) for value in self.data.xpos[body_id, :2])
        for axis, (suffix, coordinate) in enumerate(
            (("x", pose.x), ("y", pose.y))
        ):
            joint_name = f"robot_{robot_id}_{suffix}"
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id == -1:
                raise MuJoCoModelConfigurationError(
                    f"Robot '{robot_id}' requires joint '{joint_name}' for teleport"
                )
            qpos_address = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_address] += coordinate - current_world[axis]
            self.data.qvel[self.model.jnt_dofadr[joint_id]] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def remove_payload(self, payload_id: str) -> None:
        """Fault-injection hook that removes one exact package from physics."""
        if not isinstance(payload_id, str) or not payload_id:
            raise ValueError("payload_id must be a non-empty string")
        joint_name = f"package_{payload_id}_free"
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id == -1:
            raise MuJoCoModelConfigurationError(
                f"Payload '{payload_id}' requires joint '{joint_name}'"
            )
        qpos = self.model.jnt_qposadr[joint_id]
        dof = self.model.jnt_dofadr[joint_id]
        self.data.qpos[qpos:qpos + 3] = (999.0, 999.0, 0.04)
        self.data.qpos[qpos + 3:qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[dof:dof + 6] = 0.0
        self._inactive_payload_ids.add(payload_id)
        mujoco.mj_forward(self.model, self.data)

    def get_robot_collisions(
        self, robot_ids: Iterable[str]
    ) -> Set[Tuple[str, str]]:
        """Return unique chassis crash pairs from the current contact list."""
        chassis_by_geom = {}
        for robot_id in set(robot_ids):
            geom_name = f"{robot_id}_base"
            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            if geom_id == -1:
                raise MuJoCoModelConfigurationError(
                    f"Robot '{robot_id}' requires chassis geom '{geom_name}', "
                    "but it is missing from the compiled model"
                )
            chassis_by_geom[geom_id] = robot_id

        collisions: Set[Tuple[str, str]] = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first_geom = int(contact.geom1)
            second_geom = int(contact.geom2)
            first_robot = chassis_by_geom.get(first_geom)
            second_robot = chassis_by_geom.get(second_geom)

            if first_robot is not None and second_robot is not None:
                first_id, second_id = sorted((first_robot, second_robot))
                collisions.add((first_id, second_id))
                continue

            collision_robot_id: Optional[str] = first_robot or second_robot
            if collision_robot_id is None:
                continue
            obstacle_geom = second_geom if first_robot is not None else first_geom
            obstacle_name = self._crash_obstacle_name(obstacle_geom)
            if obstacle_name is not None:
                collisions.add((collision_robot_id, obstacle_name))

        return collisions

    def _crash_obstacle_name(self, geom_id: int) -> Optional[str]:
        geom_name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        if geom_name and geom_name.startswith("wall_"):
            return geom_name

        body_id = int(self.model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        if body_name and body_name.startswith("shelf_"):
            return body_name
        if body_name and body_name.startswith("package_"):
            return body_name
        return None

    def spawn_package_physically(
        self, package_id: str, x: float, y: float, z: float
    ) -> Optional[str]:
        if not isinstance(package_id, str) or not package_id:
            raise ValueError("package_id must be a non-empty string")
        for name, value in (("spawn x", x), ("spawn y", y), ("spawn z", z)):
            require_finite(name, value)
        box_name = f"package_{package_id}"
        box_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{box_name}_free"
        )
        if box_jid == -1:
            return None
        box_qa = self.model.jnt_qposadr[box_jid]
        if package_id not in self._inactive_payload_ids:
            return None

        self.data.qpos[box_qa:box_qa + 3] = (x, y, z)
        self.data.qpos[box_qa + 3:box_qa + 7] = (1.0, 0.0, 0.0, 0.0)
        # Reset velocity
        self.data.qvel[self.model.jnt_dofadr[box_jid]:self.model.jnt_dofadr[box_jid]+6] = 0.0
        self._inactive_payload_ids.remove(package_id)
        mujoco.mj_forward(self.model, self.data)

        return box_name

    def despawn_package(
        self, payload_id: str, x: float, y: float, radius: float
    ) -> bool:
        """Despawn one exact package, never an arbitrary nearby payload."""
        if not isinstance(payload_id, str) or not payload_id:
            raise ValueError("payload_id must be a non-empty string")
        require_finite("despawn x", x)
        require_finite("despawn y", y)
        require_finite("despawn radius", radius, minimum=0.0)
        joint_name = f"package_{payload_id}_free"
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id == -1:
            raise MuJoCoModelConfigurationError(
                f"Payload '{payload_id}' requires joint '{joint_name}'"
            )
        qpos = self.model.jnt_qposadr[joint_id]
        package_x = float(self.data.qpos[qpos])
        package_y = float(self.data.qpos[qpos + 1])
        distance = ((package_x - x) ** 2 + (package_y - y) ** 2) ** 0.5
        if distance > radius:
            return False
        self.remove_payload(payload_id)
        return True
