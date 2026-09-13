"""Physical contract tests for the robot turret and telescoping arm."""

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from adapters.mujoco_controller import MuJoCoController
from adapters.mujoco_environment_adapter import MuJoCoEnvironmentAdapter
from entities.enums import CardinalDirection, ManipulatorResult
from use_cases.composite_manipulator import CompositeManipulator

WAREHOUSE_XML = Path(__file__).resolve().parents[1] / "warehouse.xml"


@pytest.fixture(scope="module")
def warehouse_model():
    return mujoco.MjModel.from_xml_path(str(WAREHOUSE_XML))


@pytest.mark.parametrize("robot_id", ["r1", "r2"])
def test_warehouse_robot_has_complete_arm(warehouse_model, robot_id):
    turret_joint = mujoco.mj_name2id(
        warehouse_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"robot_{robot_id}_turret",
    )
    extend_joint = mujoco.mj_name2id(
        warehouse_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"robot_{robot_id}_extend",
    )
    turret_actuator = mujoco.mj_name2id(
        warehouse_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        f"robot_{robot_id}_turret",
    )
    extend_actuator = mujoco.mj_name2id(
        warehouse_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        f"robot_{robot_id}_extend",
    )

    assert turret_joint >= 0
    assert extend_joint >= 0
    assert turret_actuator >= 0
    assert extend_actuator >= 0
    assert warehouse_model.jnt_type[turret_joint] == mujoco.mjtJoint.mjJNT_HINGE
    assert warehouse_model.jnt_type[extend_joint] == mujoco.mjtJoint.mjJNT_SLIDE
    assert warehouse_model.jnt_axis[turret_joint] == pytest.approx((0.0, 0.0, 1.0))
    assert warehouse_model.jnt_axis[extend_joint] == pytest.approx((1.0, 0.0, 0.0))
    assert warehouse_model.jnt_range[extend_joint] == pytest.approx((0.0, 0.5))
    assert warehouse_model.actuator_trnid[turret_actuator, 0] == turret_joint
    assert warehouse_model.actuator_trnid[extend_actuator, 0] == extend_joint
    assert mujoco.mj_name2id(
        warehouse_model,
        mujoco.mjtObj.mjOBJ_BODY,
        f"robot_{robot_id}_gripper",
    ) >= 0
    assert mujoco.mj_name2id(
        warehouse_model,
        mujoco.mjtObj.mjOBJ_SITE,
        f"robot_{robot_id}_grip_site",
    ) >= 0


def test_warehouse_has_no_unused_gripper_welds(warehouse_model):
    equality_names = {
        mujoco.mj_id2name(
            warehouse_model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id
        )
        for equality_id in range(warehouse_model.neq)
    }

    assert not {
        name for name in equality_names if name and "_grip_" in name
    }


def test_joint_feedback_treats_positive_and_negative_pi_as_same_turret_angle(
    warehouse_model,
):
    data = mujoco.MjData(warehouse_model)
    turret = mujoco.mj_name2id(
        warehouse_model, mujoco.mjtObj.mjOBJ_JOINT, "robot_r1_turret"
    )
    data.qpos[warehouse_model.jnt_qposadr[turret]] = -3.141592653589793
    controller = MuJoCoController(warehouse_model, data)

    assert controller.joint_at_target(
        "r1", "turret", 3.141592653589793, 0.001
    )


@pytest.mark.parametrize(
    ("angle", "expected_direction"),
    [
        (0.0, (1.0, 0.0)),
        (3.141592653589793, (-1.0, 0.0)),
        (1.5707963267948966, (0.0, 1.0)),
        (-1.5707963267948966, (0.0, -1.0)),
    ],
)
def test_extended_grip_site_faces_each_world_direction(
    warehouse_model,
    angle,
    expected_direction,
):
    data = mujoco.MjData(warehouse_model)
    turret_joint = mujoco.mj_name2id(
        warehouse_model, mujoco.mjtObj.mjOBJ_JOINT, "robot_r1_turret"
    )
    extend_joint = mujoco.mj_name2id(
        warehouse_model, mujoco.mjtObj.mjOBJ_JOINT, "robot_r1_extend"
    )
    turret_qpos = warehouse_model.jnt_qposadr[turret_joint]
    extend_qpos = warehouse_model.jnt_qposadr[extend_joint]
    data.qpos[turret_qpos] = angle
    data.qpos[extend_qpos] = 0.5
    mujoco.mj_forward(warehouse_model, data)

    pivot_id = mujoco.mj_name2id(
        warehouse_model, mujoco.mjtObj.mjOBJ_BODY, "robot_r1_arm"
    )
    grip_site_id = mujoco.mj_name2id(
        warehouse_model, mujoco.mjtObj.mjOBJ_SITE, "robot_r1_grip_site"
    )
    displacement = data.site_xpos[grip_site_id, :2] - data.xpos[pivot_id, :2]

    assert float((displacement**2).sum() ** 0.5) == pytest.approx(0.95)
    unit_direction = displacement / float((displacement**2).sum() ** 0.5)
    assert unit_direction == pytest.approx(expected_direction, abs=1e-6)


def test_real_actuators_complete_positive_y_pick(warehouse_model):
    data = mujoco.MjData(warehouse_model)

    robot_x = mujoco.mj_name2id(
        warehouse_model, mujoco.mjtObj.mjOBJ_JOINT, "robot_r1_x"
    )
    robot_y = mujoco.mj_name2id(
        warehouse_model, mujoco.mjtObj.mjOBJ_JOINT, "robot_r1_y"
    )
    package = mujoco.mj_name2id(
        warehouse_model, mujoco.mjtObj.mjOBJ_JOINT, "package_p2_free"
    )
    data.qpos[warehouse_model.jnt_qposadr[robot_x]] = 1.0
    data.qpos[warehouse_model.jnt_qposadr[robot_y]] = -8.0
    package_qpos = warehouse_model.jnt_qposadr[package]
    data.qpos[package_qpos:package_qpos + 3] = (2.0, 2.0, 0.46)
    mujoco.mj_forward(warehouse_model, data)

    controller = MuJoCoController(warehouse_model, data)
    environment = MuJoCoEnvironmentAdapter(warehouse_model, data)
    manipulator = CompositeManipulator(
        controller,
        controller,
        controller,
        controller,
        controller,
        environment,
    )

    result = ManipulatorResult.IN_PROGRESS
    for _ in range(40):
        result = manipulator.pick("r1", "p2", 0, CardinalDirection.POS_Y)
        for _ in range(10):
            mujoco.mj_step(warehouse_model, data)
            controller.sync_attachments()
        if result is ManipulatorResult.SUCCEEDED:
            break

    assert result is ManipulatorResult.SUCCEEDED
    assert controller.attached_packages == {"r1": "package_p2_free"}
    package_position = data.qpos[package_qpos:package_qpos + 3]
    assert package_position[:2] != pytest.approx((2.0, 2.0), abs=0.25)
