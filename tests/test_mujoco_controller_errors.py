"""Regression tests for MuJoCo model/controller naming mismatches."""

import logging

import pytest

mujoco = pytest.importorskip("mujoco")

from adapters.mujoco_controller import MuJoCoController
from adapters.mujoco_model_configuration_error import (
    MuJoCoModelConfigurationError,
)
from entities.drive_command import DriveCommand
from entities.enums import CardinalDirection

MODEL_WITHOUT_ACTUATORS = """
<mujoco>
  <worldbody>
    <body name="robot_r1">
      <joint name="robot_r1_x" type="slide" axis="1 0 0"/>
      <joint name="robot_r1_y" type="slide" axis="0 1 0"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body name="robot_r1_arm">
        <joint name="robot_r1_lift" type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.05" mass="0.1"/>
        <body name="robot_r1_turret_body">
          <joint name="robot_r1_turret" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.04" mass="0.1"/>
          <body name="robot_r1_gripper">
            <joint name="robot_r1_extend" type="slide" axis="1 0 0"/>
            <geom type="sphere" size="0.03" mass="0.1"/>
            <site name="robot_r1_grip_site"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def controller_without_actuators():
    model = mujoco.MjModel.from_xml_string(MODEL_WITHOUT_ACTUATORS)
    return MuJoCoController(model, mujoco.MjData(model))


@pytest.mark.parametrize(
    ("method_name", "arguments", "missing_name"),
    [
        (
            "command_velocity",
            ("r1", DriveCommand(0.1, 0.0)),
            "robot_r1_x",
        ),
        ("stop", ("r1",), "robot_r1_x"),
        ("set_lift_height", ("r1", 0.38), "robot_r1_lift"),
        ("extend_arm", ("r1", 0.5), "robot_r1_extend"),
        (
            "aim_turret",
            ("r1", CardinalDirection.POS_X),
            "robot_r1_turret",
        ),
        (
            "joint_at_target",
            ("r1", "missing", 0.0, 0.01),
            "robot_r1_missing",
        ),
    ],
)
def test_missing_actuator_is_logged_and_raised(
    controller_without_actuators,
    method_name,
    arguments,
    missing_name,
    caplog,
):
    with (
        caplog.at_level(logging.ERROR, logger="adapters.mujoco_controller"),
        pytest.raises(MuJoCoModelConfigurationError, match=missing_name),
    ):
        getattr(controller_without_actuators, method_name)(*arguments)

    assert "r1" in caplog.text
    assert missing_name in caplog.text


def test_unexpected_lookup_error_is_not_swallowed(
    controller_without_actuators,
    monkeypatch,
):
    def fail_lookup(*_args):
        raise TypeError("invalid compiled model")

    monkeypatch.setattr(mujoco, "mj_name2id", fail_lookup)

    with pytest.raises(TypeError, match="invalid compiled model"):
        controller_without_actuators.stop("r1")


def test_joint_feedback_rejects_negative_tolerance(
    controller_without_actuators,
):
    with pytest.raises(ValueError, match="non-negative"):
        controller_without_actuators.joint_at_target(
            "r1", "lift", 0.0, -0.01
        )


def test_grip_rejects_missing_payload_joint(controller_without_actuators):
    with pytest.raises(MuJoCoModelConfigurationError, match="package_p1_free"):
        controller_without_actuators.grip("r1", "p1", engage=True)


def test_grip_rejects_missing_robot_site(controller_without_actuators):
    with pytest.raises(MuJoCoModelConfigurationError, match="robot_ghost_grip_site"):
        controller_without_actuators.grip("ghost", "p1", engage=True)
