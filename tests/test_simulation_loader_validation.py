"""Fail-fast checks for logical/physical warehouse model agreement."""

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from adapters.mujoco_model_configuration_error import (
    MuJoCoModelConfigurationError,
)
from adapters.simulation_loader import SimulationLoader

WAREHOUSE_XML = Path(__file__).resolve().parents[1] / "warehouse.xml"


def _load_modified_model(tmp_path, old, new):
    xml = WAREHOUSE_XML.read_text()
    assert old in xml, f"test setup could not find {old!r}"
    path = tmp_path / "warehouse_modified.xml"
    path.write_text(xml.replace(old, new, 1))
    return SimulationLoader(str(path)).load()


def test_loader_accepts_complete_sensor_model():
    model, data, topology = SimulationLoader(str(WAREHOUSE_XML)).load()

    assert model.nsensor == 7 * 4
    assert data is not None
    assert topology.max_x == 10.0


def test_loader_rejects_a_missing_sensor(tmp_path):
    with pytest.raises(MuJoCoModelConfigurationError, match="wall_pos_x"):
        _load_modified_model(
            tmp_path,
            'name="robot_r1_wall_pos_x"',
            'name="robot_r1_wall_pos_x_missing"',
        )


def test_loader_rejects_a_missing_arm_actuator(tmp_path):
    with pytest.raises(MuJoCoModelConfigurationError, match="robot_r1_extend"):
        _load_modified_model(
            tmp_path,
            'name="robot_r1_extend" kp="1000"',
            'name="robot_r1_extend_missing" kp="1000"',
        )


def test_loader_rejects_a_missing_grip_site(tmp_path):
    with pytest.raises(MuJoCoModelConfigurationError, match="grip_site"):
        _load_modified_model(
            tmp_path,
            'name="robot_r1_grip_site"',
            'name="robot_r1_grip_site_missing"',
        )


def test_loader_rejects_a_misdirected_wall_ray(tmp_path):
    with pytest.raises(MuJoCoModelConfigurationError, match="direction"):
        _load_modified_model(
            tmp_path,
            'name="robot_r1_wall_pos_x_site" pos="0 0 0.88" zaxis="1 0 0"',
            'name="robot_r1_wall_pos_x_site" pos="0 0 0.88" zaxis="0 1 0"',
        )


def test_loader_rejects_wall_ray_at_shelf_height(tmp_path):
    with pytest.raises(MuJoCoModelConfigurationError, match="world height"):
        _load_modified_model(
            tmp_path,
            'name="robot_r1_wall_pos_x_site" pos="0 0 0.88"',
            'name="robot_r1_wall_pos_x_site" pos="0 0 0.60"',
        )


def test_loader_rejects_wrong_perimeter_bound(tmp_path):
    with pytest.raises(MuJoCoModelConfigurationError, match="wall_east"):
        _load_modified_model(
            tmp_path,
            'name="wall_east" type="box" size="0.05 5.05 0.5" pos="10.05 5 0.5"',
            'name="wall_east" type="box" size="0.05 5.05 0.5" pos="10.25 5 0.5"',
        )


def test_loader_rejects_logical_shelf_moved_in_xml(tmp_path):
    with pytest.raises(MuJoCoModelConfigurationError, match="shelf_2_2"):
        _load_modified_model(
            tmp_path,
            'body name="shelf_2_2" pos="2 2 0"',
            'body name="shelf_2_2" pos="2.2 2 0"',
        )


def test_loader_rejects_logical_zone_moved_in_xml(tmp_path):
    with pytest.raises(MuJoCoModelConfigurationError, match="zone_inbound"):
        _load_modified_model(
            tmp_path,
            'name="zone_inbound" type="box" size="0.5 0.5 0.005" pos="9.0 1.0 0.005"',
            'name="zone_inbound" type="box" size="0.5 0.5 0.005" pos="8.0 1.0 0.005"',
        )


def test_all_shelf_and_zone_geoms_have_stable_names():
    model, _, _ = SimulationLoader(str(WAREHOUSE_XML)).load()
    unnamed_shelf_geoms = []
    for geom_id in range(model.ngeom):
        body_name = model.body(model.geom_bodyid[geom_id]).name
        geom_name = model.geom(geom_id).name
        if body_name.startswith("shelf_") and not geom_name:
            unnamed_shelf_geoms.append(geom_id)

    assert unnamed_shelf_geoms == []
    assert model.geom("zone_parking").id >= 0
    assert model.geom("zone_inbound").id >= 0
    assert model.geom("zone_outbound").id >= 0
