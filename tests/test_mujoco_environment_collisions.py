"""Focused tests for physical collision KPI filtering."""

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from adapters.mujoco_environment_adapter import MuJoCoEnvironmentAdapter
from adapters.mujoco_model_configuration_error import (
    MuJoCoModelConfigurationError,
)

CONTACT_MODEL_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="12 12 0.1"/>
    <geom name="wall_west" type="box" pos="0 0 0.2" size="0.1 1 0.2"/>
    <body name="shelf_2_2" pos="2 0 0.2">
      <geom name="shelf_2_2_a" type="box" size="0.2 0.2 0.2"/>
      <geom name="shelf_2_2_b" type="box" size="0.2 0.2 0.2"/>
    </body>
    <body name="package_p1" pos="4 0 0.1">
      <freejoint name="package_p1_free"/>
      <geom name="package_p1_geom" type="box" size="0.1 0.1 0.1"/>
    </body>
    <body name="robot_r1" pos="8 0 0.1">
      <freejoint name="robot_r1_free"/>
      <geom name="r1_base" type="cylinder" size="0.2 0.1"/>
      <body name="robot_r1_gripper" pos="0.6 0 0">
        <geom name="r1_gripper_plate" type="box" size="0.1 0.1 0.1"/>
      </body>
    </body>
    <body name="robot_r2" pos="9 0 0.1">
      <freejoint name="robot_r2_free"/>
      <geom name="r2_base" type="cylinder" size="0.2 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _environment():
    model = mujoco.MjModel.from_xml_string(CONTACT_MODEL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return MuJoCoEnvironmentAdapter(model, data)


def _move_free_body(environment, joint_name, x, y):
    joint_id = mujoco.mj_name2id(
        environment.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
    )
    qpos_address = environment.model.jnt_qposadr[joint_id]
    environment.data.qpos[qpos_address] = x
    environment.data.qpos[qpos_address + 1] = y
    mujoco.mj_forward(environment.model, environment.data)


def test_floor_support_contact_is_not_an_actual_collision():
    environment = _environment()

    assert environment.get_robot_collisions(["r1", "r2"]) == set()


def test_registered_robot_without_chassis_geom_fails_loudly():
    environment = _environment()

    with pytest.raises(MuJoCoModelConfigurationError, match="ghost_base"):
        environment.get_robot_collisions(["ghost"])


def test_robot_chassis_collision_is_reported_once_for_the_pair():
    environment = _environment()
    _move_free_body(environment, "robot_r1_free", 6.0, 0.0)
    _move_free_body(environment, "robot_r2_free", 6.25, 0.0)

    assert environment.get_robot_collisions(["r2", "r1"]) == {
        ("r1", "r2")
    }


def test_chassis_wall_collision_is_reported():
    environment = _environment()
    _move_free_body(environment, "robot_r1_free", 0.25, 0.0)

    assert environment.get_robot_collisions(["r1"]) == {
        ("r1", "wall_west")
    }


def test_multiple_chassis_shelf_contacts_are_deduplicated():
    environment = _environment()
    _move_free_body(environment, "robot_r1_free", 1.75, 0.0)

    assert environment.get_robot_collisions(["r1"]) == {
        ("r1", "shelf_2_2")
    }


def test_chassis_package_collision_is_reported():
    environment = _environment()
    _move_free_body(environment, "robot_r1_free", 3.75, 0.0)

    assert environment.get_robot_collisions(["r1"]) == {
        ("r1", "package_p1")
    }


@pytest.mark.parametrize(
    ("robot_x", "expected_contact"),
    [(1.4, "shelf"), (3.4, "package")],
)
def test_intended_gripper_contacts_are_excluded(robot_x, expected_contact):
    environment = _environment()
    _move_free_body(environment, "robot_r1_free", robot_x, 0.0)
    assert environment.data.ncon > 0, f"test setup produced no {expected_contact} contact"

    assert environment.get_robot_collisions(["r1"]) == set()


def test_warehouse_perimeter_walls_have_stable_collision_names():
    xml_path = Path(__file__).parents[1] / "warehouse.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    wall_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(model.ngeom)
    }
    assert {
        "wall_south",
        "wall_north",
        "wall_west",
        "wall_east",
    } <= wall_names
