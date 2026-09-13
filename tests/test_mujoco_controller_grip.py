"""Tests for the MuJoCo gripper adapter.

The rest of the suite runs without MuJoCo installed. This module needs it,
because the bug it guards against was a mismatch between the code's assumed
joint names and the names actually present in a compiled model.
"""
import pytest

mujoco = pytest.importorskip("mujoco")

from adapters.mujoco_controller import MuJoCoController
from entities.enums import CardinalDirection

MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="robot_r1_arm" pos="1 1 0.5">
      <geom type="box" size="0.1 0.1 0.02"/>
      <site name="robot_r1_grip_site" type="sphere" size="0.01"/>
    </body>
    <body name="package_p1" pos="1 1 0.6">
      <joint type="free" name="package_p1_free"/>
      <geom type="box" size="0.05 0.05 0.04"/>
    </body>
    <body name="package_p2" pos="8 8 0.5">
      <joint type="free" name="package_p2_free"/>
      <geom type="box" size="0.05 0.05 0.04"/>
    </body>
  </worldbody>
</mujoco>
"""


ARM_MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="robot_r1_arm" pos="1 1 0.5">
      <body name="robot_r1_turret_body">
        <joint type="hinge" name="robot_r1_turret" axis="0 0 1"/>
        <geom type="sphere" size="0.01" mass="0.1"/>
        <body name="robot_r1_gripper" pos="0.45 0 0">
          <joint type="slide" name="robot_r1_extend" axis="1 0 0" range="0 0.5"/>
          <geom type="sphere" size="0.01" mass="0.1"/>
          <site name="robot_r1_grip_site" type="sphere" size="0.01"/>
        </body>
      </body>
    </body>
    <body name="package_at_tip" pos="1.95 1 0.6">
      <joint type="free" name="package_at_tip_free"/>
      <geom type="box" size="0.05 0.05 0.04"/>
    </body>
    <body name="package_at_root" pos="1 1 0.6">
      <joint type="free" name="package_at_root_free"/>
      <geom type="box" size="0.05 0.05 0.04"/>
    </body>
  </worldbody>
  <actuator>
    <position name="robot_r1_turret" joint="robot_r1_turret"/>
    <position name="robot_r1_extend" joint="robot_r1_extend"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def controller():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return MuJoCoController(model, data)


def _package_pos(controller, joint_name):
    jnt = mujoco.mj_name2id(controller.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr = controller.model.jnt_qposadr[jnt]
    return tuple(float(v) for v in controller.data.qpos[adr:adr + 3])


def test_package_carry_offset_is_a_named_physical_calibration():
    assert pytest.approx(0.1) == MuJoCoController.PACKAGE_CARRY_OFFSET_METERS


def test_grip_attaches_the_nearest_package():
    """Regression test: grip() searched for joints prefixed 'pkg_', but the
    model names them 'package_<id>_free'. The search never matched, so no
    package was ever attached and the gripper was silently a no-op.
    """
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctrl = MuJoCoController(model, data)

    ctrl.grip("r1", "p1", engage=True)

    assert ctrl.attached_packages == {"r1": "package_p1_free"}


def test_grip_ignores_packages_out_of_reach(controller):
    controller.grip("r1", "p2", engage=True)
    assert controller.attached_packages.get("r1") != "package_p2_free"


def test_release_detaches(controller):
    controller.grip("r1", "p1", engage=True)
    assert "r1" in controller.attached_packages

    controller.grip("r1", "p1", engage=False)
    assert "r1" not in controller.attached_packages


def test_sync_attachments_carries_the_package_with_the_arm(controller):
    """Attaching is only half the job: the package has to follow the arm.

    The package is displaced after being gripped, so this can only pass if
    sync_attachments actually repositions it rather than leaving it be.
    """
    controller.grip("r1", "p1", engage=True)

    jnt = mujoco.mj_name2id(controller.model, mujoco.mjtObj.mjOBJ_JOINT, "package_p1_free")
    adr = controller.model.jnt_qposadr[jnt]
    controller.data.qpos[adr:adr + 3] = [5.0, 5.0, 2.0]

    controller.sync_attachments()

    x, y, z = _package_pos(controller, "package_p1_free")
    assert (x, y) == pytest.approx((1.0, 1.0))
    assert z == pytest.approx(0.6)  # arm z 0.5 plus the 0.1 carry offset


def test_sync_attachments_does_nothing_when_empty_handed(controller):
    before = _package_pos(controller, "package_p1_free")
    controller.sync_attachments()
    assert _package_pos(controller, "package_p1_free") == pytest.approx(before)


def test_carried_package_does_not_accumulate_velocity(controller):
    """A carried package is positioned by the gripper, not by physics.

    Without zeroing its velocity it keeps accelerating under gravity between
    teleports, and shoots away the moment it is released.
    """
    controller.grip("r1", "p1", engage=True)
    for _ in range(50):
        mujoco.mj_step(controller.model, controller.data)
        controller.sync_attachments()

    jnt = mujoco.mj_name2id(controller.model, mujoco.mjtObj.mjOBJ_JOINT, "package_p1_free")
    adr = controller.model.jnt_dofadr[jnt]
    velocity = controller.data.qvel[adr:adr + 6]
    assert max(abs(float(v)) for v in velocity) < 1e-6


@pytest.mark.parametrize(
    ("direction", "expected_angle"),
    [
        (CardinalDirection.POS_X, 0.0),
        (CardinalDirection.NEG_X, 3.141592653589793),
        (CardinalDirection.POS_Y, 1.5707963267948966),
        (CardinalDirection.NEG_Y, -1.5707963267948966),
    ],
)
def test_aim_turret_maps_cardinal_direction_to_actuator(direction, expected_angle):
    model = mujoco.MjModel.from_xml_string(ARM_MODEL_XML)
    data = mujoco.MjData(model)
    controller = MuJoCoController(model, data)

    controller.aim_turret("r1", direction)

    actuator = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot_r1_turret"
    )
    assert data.ctrl[actuator] == pytest.approx(expected_angle)


@pytest.mark.parametrize(
    ("direction", "angle", "tip_xy"),
    [
        (CardinalDirection.POS_X, 0.0, (1.95, 1.0)),
        (CardinalDirection.NEG_X, 3.141592653589793, (0.05, 1.0)),
        (CardinalDirection.POS_Y, 1.5707963267948966, (1.0, 1.95)),
        (CardinalDirection.NEG_Y, -1.5707963267948966, (1.0, 0.05)),
    ],
)
def test_grip_uses_extended_tip_in_every_direction(direction, angle, tip_xy):
    model = mujoco.MjModel.from_xml_string(ARM_MODEL_XML)
    data = mujoco.MjData(model)
    turret_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "robot_r1_turret"
    )
    extend_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "robot_r1_extend"
    )
    package_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "package_at_tip_free"
    )
    data.qpos[model.jnt_qposadr[turret_joint]] = angle
    data.qpos[model.jnt_qposadr[extend_joint]] = 0.5
    package_qpos = model.jnt_qposadr[package_joint]
    data.qpos[package_qpos:package_qpos + 3] = [*tip_xy, 0.6]
    mujoco.mj_forward(model, data)
    controller = MuJoCoController(model, data)

    controller.aim_turret("r1", direction)
    controller.grip("r1", "at_tip", engage=True)

    assert controller.attached_packages == {"r1": "package_at_tip_free"}


def test_retracted_gripper_cannot_pick_from_an_adjacent_station():
    model = mujoco.MjModel.from_xml_string(ARM_MODEL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    controller = MuJoCoController(model, data)

    controller.grip("r1", "at_tip", engage=True)

    assert controller.attached_packages == {}
