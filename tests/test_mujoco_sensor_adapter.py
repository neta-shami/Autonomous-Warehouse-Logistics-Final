"""Physical sensor and ground-truth adapter contract tests."""

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from adapters.mujoco_ground_truth_probe import MuJoCoGroundTruthProbe
from adapters.sensor_adapter import MuJoCoSensorAdapter
from entities.enums import CardinalDirection
from entities.pose import Pose
from interfaces.sensor_read_error import SensorReadError

WAREHOUSE_XML = Path(__file__).resolve().parents[1] / "warehouse.xml"


@pytest.fixture()
def warehouse():
    model = mujoco.MjModel.from_xml_path(str(WAREHOUSE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _set_joint(model, data, name, value, *, velocity=None):
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    data.qpos[model.jnt_qposadr[joint_id]] = value
    if velocity is not None:
        data.qvel[model.jnt_dofadr[joint_id]] = velocity


def _move_r1_to_fractional_pose(model, data):
    # Slide qpos is relative to the body's XML position (1, 9).
    _set_joint(model, data, "robot_r1_x", 2.25, velocity=0.4)
    _set_joint(model, data, "robot_r1_y", -4.5, velocity=-0.3)
    mujoco.mj_forward(model, data)


def test_reads_fractional_wall_fix_and_joint_velocity(warehouse):
    model, data = warehouse
    _move_r1_to_fractional_pose(model, data)
    adapter = MuJoCoSensorAdapter(model, data, range_noise_bound_m=0.0)

    frame = adapter.read_frame("r1")

    assert dict(frame.wall_ranges) == pytest.approx(
        {
            CardinalDirection.POS_X: 6.75,
            CardinalDirection.NEG_X: 3.25,
            CardinalDirection.POS_Y: 5.5,
            CardinalDirection.NEG_Y: 4.5,
        }
    )
    assert frame.observed_chassis_velocity == pytest.approx((0.4, -0.3))
    assert frame.payload_range == -1.0
    assert frame.sim_time == pytest.approx(0.0)
    assert frame.sequence == 0


def test_sequences_are_per_robot_and_timestamps_must_advance(warehouse):
    model, data = warehouse
    adapter = MuJoCoSensorAdapter(model, data, range_noise_bound_m=0.0)

    assert adapter.read_frame("r1").sequence == 0
    assert adapter.read_frame("r2").sequence == 0
    with pytest.raises(SensorReadError, match="did not advance"):
        adapter.read_frame("r1")

    data.time = 0.02
    assert adapter.read_frame("r1").sequence == 1


def test_wrong_robot_is_rejected_instead_of_cross_wired(warehouse):
    model, data = warehouse
    adapter = MuJoCoSensorAdapter(model, data)

    with pytest.raises(SensorReadError, match="ghost"):
        adapter.read_frame("ghost")


def test_non_finite_reading_is_contained_without_advancing_history(warehouse):
    model, data = warehouse
    sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "robot_r1_wall_pos_x"
    )
    address = model.sensor_adr[sensor_id]
    healthy_value = float(data.sensordata[address])
    data.sensordata[address] = float("nan")
    adapter = MuJoCoSensorAdapter(model, data, range_noise_bound_m=0.0)

    with pytest.raises(SensorReadError, match="non-finite"):
        adapter.read_frame("r1")

    data.sensordata[address] = healthy_value
    assert adapter.read_frame("r1").sequence == 0


def test_seeded_noise_is_reproducible_and_bounded(warehouse):
    model, data = warehouse
    _move_r1_to_fractional_pose(model, data)
    first = MuJoCoSensorAdapter(
        model, data, random_seed=17, range_noise_bound_m=0.01
    ).read_frame("r1")
    second = MuJoCoSensorAdapter(
        model, data, random_seed=17, range_noise_bound_m=0.01
    ).read_frame("r1")

    assert first == second
    expected = {
        CardinalDirection.POS_X: 6.75,
        CardinalDirection.NEG_X: 3.25,
        CardinalDirection.POS_Y: 5.5,
        CardinalDirection.NEG_Y: 4.5,
    }
    for direction, reading in first.wall_ranges:
        assert abs(reading - expected[direction]) <= 0.01


def test_healthy_no_return_is_not_corrupted_by_noise(warehouse):
    model, data = warehouse
    # Move beyond the east wall so the +X ray has no hit.
    _set_joint(model, data, "robot_r1_x", 10.0)
    mujoco.mj_forward(model, data)
    adapter = MuJoCoSensorAdapter(
        model, data, random_seed=4, range_noise_bound_m=0.5
    )

    assert dict(adapter.read_frame("r1").wall_ranges)[
        CardinalDirection.POS_X
    ] == -1.0


def test_payload_up_ray_detects_a_carried_package(warehouse):
    model, data = warehouse
    grip_site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "robot_r1_grip_site"
    )
    package_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "package_p1_free"
    )
    package_qpos = model.jnt_qposadr[package_joint]
    grip_position = data.site_xpos[grip_site]
    data.qpos[package_qpos:package_qpos + 3] = (
        grip_position[0],
        grip_position[1],
        grip_position[2] + 0.1,
    )
    mujoco.mj_forward(model, data)

    frame = MuJoCoSensorAdapter(
        model, data, range_noise_bound_m=0.0
    ).read_frame("r1")

    assert frame.payload_range == pytest.approx(0.02, abs=1e-6)


@pytest.mark.parametrize(
    ("lift", "turret", "extension"),
    [
        (0.0, 0.0, 0.0),
        (0.0, 3.141592653589793, 0.5),
        (1.0, -3.141592653589793, 0.0),
        (1.0, 1.5707963267948966, 0.5),
    ],
)
def test_arm_extremes_do_not_occlude_wall_ranges(
    warehouse, lift, turret, extension
):
    model, data = warehouse
    _move_r1_to_fractional_pose(model, data)
    _set_joint(model, data, "robot_r1_lift", lift)
    _set_joint(model, data, "robot_r1_turret", turret)
    _set_joint(model, data, "robot_r1_extend", extension)
    mujoco.mj_forward(model, data)

    frame = MuJoCoSensorAdapter(
        model, data, range_noise_bound_m=0.0
    ).read_frame("r1")

    assert tuple(distance for _, distance in frame.wall_ranges) == pytest.approx(
        (6.75, 3.25, 5.5, 4.5)
    )


def test_passing_robot_below_sensor_plane_does_not_occlude_fix(warehouse):
    model, data = warehouse
    _move_r1_to_fractional_pose(model, data)
    # Put r2 directly between r1 and the east wall at the same Y coordinate.
    _set_joint(model, data, "robot_r2_x", -4.0)
    _set_joint(model, data, "robot_r2_y", -4.5)
    mujoco.mj_forward(model, data)

    ranges = dict(
        MuJoCoSensorAdapter(
            model, data, range_noise_bound_m=0.0
        ).read_frame("r1").wall_ranges
    )

    assert ranges[CardinalDirection.POS_X] == pytest.approx(6.75)


def test_ground_truth_probe_is_separate_and_rejects_unknown_robot(warehouse):
    model, data = warehouse
    _move_r1_to_fractional_pose(model, data)
    probe = MuJoCoGroundTruthProbe(model, data)

    assert probe.true_pose("r1") == Pose(3.25, 4.5)
    with pytest.raises(SensorReadError, match="ghost"):
        probe.true_pose("ghost")
