import logging
from typing import Dict, Set

import mujoco
import numpy as np

from adapters.mujoco_model_configuration_error import (
    MuJoCoModelConfigurationError,
)
from entities.drive_command import DriveCommand
from entities.enums import CardinalDirection
from interfaces.i_attachment_synchronizer import IAttachmentSynchronizer
from interfaces.i_drive_system import IDriveSystem
from interfaces.i_extend_system import IExtendSystem
from interfaces.i_grip_system import IGripSystem
from interfaces.i_lift_system import ILiftSystem
from interfaces.i_manipulator_feedback import IManipulatorFeedback
from interfaces.i_turret_system import ITurretSystem

LOGGER = logging.getLogger(__name__)


class MuJoCoController(
    IDriveSystem,
    ILiftSystem,
    IExtendSystem,
    IGripSystem,
    ITurretSystem,
    IManipulatorFeedback,
    IAttachmentSynchronizer,
):
    """
    Implements the drive, turret, lift, extension and gripper hardware ports.
    Translates logic decisions into actuator forces and physics updates.
    """

    # Must match the free-joint naming in warehouse.xml ("package_<id>_free").
    PACKAGE_JOINT_PREFIX = "package_"
    GRIP_REACH_METERS = 0.25
    PACKAGE_CARRY_OFFSET_METERS = 0.1

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.attached_packages: Dict[str, str] = {}
        self._stalled_robot_ids: Set[str] = set()

    def _require_model_element(
        self,
        object_type,
        element_kind: str,
        element_name: str,
        robot_id: str,
    ) -> int:
        """Resolve a required named element or report a model/code mismatch."""
        element_id = mujoco.mj_name2id(
            self.model, object_type, element_name
        )
        if element_id == -1:
            message = (
                f"Robot '{robot_id}' requires MuJoCo {element_kind} "
                f"'{element_name}', but it is missing from the compiled model"
            )
            LOGGER.error(message)
            raise MuJoCoModelConfigurationError(message)
        return element_id

    def _require_actuator(self, robot_id: str, suffix: str) -> int:
        return self._require_model_element(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "actuator",
            f"robot_{robot_id}_{suffix}",
            robot_id,
        )

    def command_velocity(self, robot_id: str, command: DriveCommand) -> None:
        """Apply the exact Layer-2-bounded X/Y command without reclamping."""
        if not isinstance(command, DriveCommand):
            raise TypeError("command must be a DriveCommand")
        act_x = self._require_actuator(robot_id, "x")
        act_y = self._require_actuator(robot_id, "y")
        # Motor gain and slide damping are calibrated 1:1 at steady state.
        if robot_id in self._stalled_robot_ids:
            self.data.ctrl[act_x] = 0.0
            self.data.ctrl[act_y] = 0.0
        else:
            self.data.ctrl[act_x] = command.vx
            self.data.ctrl[act_y] = command.vy

    def set_drive_stalled(self, robot_id: str, stalled: bool) -> None:
        """Fault-injection hook; normal control continues recording its intent."""
        if not isinstance(stalled, bool):
            raise TypeError("stalled must be bool")
        self._require_actuator(robot_id, "x")
        self._require_actuator(robot_id, "y")
        if stalled:
            self._stalled_robot_ids.add(robot_id)
            self.stop(robot_id)
        else:
            self._stalled_robot_ids.discard(robot_id)

    def stop(self, robot_id: str) -> None:
        act_x = self._require_actuator(robot_id, "x")
        act_y = self._require_actuator(robot_id, "y")
        self.data.ctrl[act_x] = 0.0
        self.data.ctrl[act_y] = 0.0

    def set_lift_height(self, robot_id: str, target_z: float) -> None:
        actuator_id = self._require_actuator(robot_id, "lift")
        self.data.ctrl[actuator_id] = target_z

    def extend_arm(self, robot_id: str, extension: float) -> None:
        actuator_id = self._require_actuator(robot_id, "extend")
        self.data.ctrl[actuator_id] = extension

    def aim_turret(
        self, robot_id: str, approach: CardinalDirection
    ) -> None:
        actuator_id = self._require_actuator(robot_id, "turret")
        self.data.ctrl[actuator_id] = approach.angle_radians

    def joint_at_target(
        self,
        robot_id: str,
        joint_name: str,
        target: float,
        tolerance: float,
    ) -> bool:
        if tolerance < 0.0:
            raise ValueError("Joint tolerance must be non-negative")
        model_name = f"robot_{robot_id}_{joint_name}"
        joint_id = self._require_model_element(
            mujoco.mjtObj.mjOBJ_JOINT,
            "joint",
            model_name,
            robot_id,
        )
        qpos_address = self.model.jnt_qposadr[joint_id]
        actual = float(self.data.qpos[qpos_address])
        error = actual - target
        if joint_name == "turret":
            error = float(np.arctan2(np.sin(error), np.cos(error)))
        return abs(error) <= tolerance

    def grip(
        self, robot_id: str, expected_payload_id: str, engage: bool
    ) -> None:
        if not isinstance(expected_payload_id, str) or not expected_payload_id:
            raise ValueError("expected_payload_id must be a non-empty string")
        expected_joint = f"{self.PACKAGE_JOINT_PREFIX}{expected_payload_id}_free"
        if engage:
            grip_site_id = self._require_model_element(
                mujoco.mjtObj.mjOBJ_SITE,
                "site",
                f"robot_{robot_id}_grip_site",
                robot_id,
            )
            grip_pos = self.data.site_xpos[grip_site_id]

            jnt_id = self._require_model_element(
                mujoco.mjtObj.mjOBJ_JOINT,
                "joint",
                expected_joint,
                robot_id,
            )
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            pkg_pos = self.data.qpos[qpos_adr:qpos_adr + 3]
            if float(np.linalg.norm(pkg_pos - grip_pos)) < self.GRIP_REACH_METERS:
                self.attached_packages[robot_id] = expected_joint
        else:
            if self.attached_packages.get(robot_id) == expected_joint:
                del self.attached_packages[robot_id]

    def detach_payload(self, payload_id: str) -> bool:
        """Remove every logical attachment to one physically stolen payload."""
        if not isinstance(payload_id, str) or not payload_id:
            raise ValueError("payload_id must be a non-empty string")
        expected_joint = f"{self.PACKAGE_JOINT_PREFIX}{payload_id}_free"
        carriers = tuple(
            robot_id
            for robot_id, joint_name in self.attached_packages.items()
            if joint_name == expected_joint
        )
        for robot_id in carriers:
            del self.attached_packages[robot_id]
        return bool(carriers)

    def sync_attachments(self) -> None:
        """Holds every gripped package at its carrier's arm.

        Must be called once per tick, otherwise a package is attached logically
        but left behind physically.
        """
        for robot_id, pkg_name in self.attached_packages.items():
            grip_site_id = self._require_model_element(
                mujoco.mjtObj.mjOBJ_SITE,
                "site",
                f"robot_{robot_id}_grip_site",
                robot_id,
            )

            jnt_id = self._require_model_element(
                mujoco.mjtObj.mjOBJ_JOINT,
                "joint",
                pkg_name,
                robot_id,
            )
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            grip_pos = self.data.site_xpos[grip_site_id]
            self.data.qpos[qpos_adr] = grip_pos[0]
            self.data.qpos[qpos_adr+1] = grip_pos[1]
            self.data.qpos[qpos_adr+2] = (
                grip_pos[2] + self.PACKAGE_CARRY_OFFSET_METERS
            )

            # A carried package is placed by the gripper, not by physics.
            # Left alone, gravity keeps building velocity between teleports
            # and the package shoots off the instant it is released.
            dof_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dof_adr:dof_adr + 6] = 0.0
