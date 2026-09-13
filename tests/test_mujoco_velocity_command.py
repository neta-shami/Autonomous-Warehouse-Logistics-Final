"""MuJoCo boundary tests for already-bounded drive commands."""

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from adapters.mujoco_controller import MuJoCoController
from adapters.simulation_loader import SimulationLoader
from entities.drive_command import DriveCommand
from entities.pose import Pose
from use_cases.waypoint_controller import WaypointController

WAREHOUSE_XML = Path(__file__).resolve().parents[1] / "warehouse.xml"


def _control(model, data, name):
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
    )
    return float(data.ctrl[actuator_id])


def test_hardware_applies_exact_bounded_command_without_second_clamp():
    model, data, _ = SimulationLoader(str(WAREHOUSE_XML)).load()
    hardware = MuJoCoController(model, data)
    command = DriveCommand(0.6, -0.8)

    hardware.command_velocity("r1", command)

    assert _control(model, data, "robot_r1_x") == pytest.approx(command.vx)
    assert _control(model, data, "robot_r1_y") == pytest.approx(command.vy)


def test_command_depends_on_belief_not_changed_physical_xpos():
    model, data, _ = SimulationLoader(str(WAREHOUSE_XML)).load()
    hardware = MuJoCoController(model, data)
    waypoint_controller = WaypointController(max_speed_mps=1.0)
    belief = Pose(1.0, 1.0)
    waypoint = (4.0, 1.0)
    first = waypoint_controller.command_for(belief, waypoint, 0.1)
    hardware.command_velocity("r1", first)

    x_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "robot_r1_x"
    )
    data.qpos[model.jnt_qposadr[x_joint]] = 7.0
    mujoco.mj_forward(model, data)
    after_physical_move = waypoint_controller.command_for(
        belief, waypoint, 0.1
    )
    hardware.command_velocity("r1", after_physical_move)

    assert after_physical_move == first
    assert _control(model, data, "robot_r1_x") == pytest.approx(first.vx)
    changed_belief = waypoint_controller.command_for(
        Pose(3.8, 1.0), waypoint, 0.1
    )
    assert changed_belief != first


def test_command_velocity_requires_drive_command():
    model, data, _ = SimulationLoader(str(WAREHOUSE_XML)).load()

    with pytest.raises(TypeError, match="DriveCommand"):
        MuJoCoController(model, data).command_velocity("r1", (0.1, 0.0))
