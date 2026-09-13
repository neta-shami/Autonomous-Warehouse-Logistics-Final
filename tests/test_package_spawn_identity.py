from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from adapters.mujoco_environment_adapter import MuJoCoEnvironmentAdapter


def _package_z(model, data, package_id):
    joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"package_{package_id}_free"
    )
    return float(data.qpos[model.jnt_qposadr[joint] + 2])


def _package_x(model, data, package_id):
    joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"package_{package_id}_free"
    )
    return float(data.qpos[model.jnt_qposadr[joint]])


def test_spawn_activates_the_requested_physical_payload_only():
    xml = Path(__file__).parents[1] / "warehouse.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    environment = MuJoCoEnvironmentAdapter(model, data)

    for _ in range(3_000):
        mujoco.mj_step(model, data)

    spawned = environment.spawn_package_physically("p5", 9.0, 1.0, 0.46)

    assert spawned == "package_p5"
    assert _package_z(model, data, "p5") == pytest.approx(0.46)
    assert _package_x(model, data, "p1") > 10.0


def test_spawn_rejects_unknown_or_already_active_payload():
    xml = Path(__file__).parents[1] / "warehouse.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    environment = MuJoCoEnvironmentAdapter(model, data)

    assert environment.spawn_package_physically("missing", 9, 1, 0.46) is None
    assert environment.spawn_package_physically("p2", 9, 1, 0.46) is None


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), True])
def test_spawn_rejects_non_finite_coordinates(coordinate):
    xml = Path(__file__).parents[1] / "warehouse.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    environment = MuJoCoEnvironmentAdapter(model, data)

    with pytest.raises((TypeError, ValueError)):
        environment.spawn_package_physically("p5", coordinate, 1.0, 0.46)
