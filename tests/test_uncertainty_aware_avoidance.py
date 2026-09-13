from entities.enums import FSMStatus, LocalizationStatus, PoseSource
from entities.fleet_snapshot import FleetRobotView, FleetSnapshot
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from use_cases.local_collision_avoidance import LocalCollisionAvoidance


def _view(robot_id, x, uncertainty=0.0, velocity=(0.0, 0.0)):
    return FleetRobotView(
        robot_id,
        PoseEstimate(Pose(x, 1.0), uncertainty, PoseSource.FUSED, 1.0),
        velocity,
        0.3,
        (),
        LocalizationStatus.TRUSTED,
        frozenset(),
        FSMStatus.NAVIGATING,
    )


def _check(first, second, **settings):
    snapshot = FleetSnapshot(1, 1.0, (first, second))
    return LocalCollisionAvoidance(**settings).check_collisions(
        first.robot_id, snapshot
    )


def test_pose_uncertainty_expands_required_separation():
    clear, _ = _check(_view("r1", 1.0), _view("r2", 1.8))
    blocked, ids = _check(
        _view("r1", 1.0, 0.1), _view("r2", 1.8, 0.1)
    )

    assert clear is None
    assert blocked is FSMStatus.AVOIDING
    assert ids == ("r2",)


def test_closing_velocity_adds_braking_distance():
    status, _ = _check(
        _view("r1", 1.0, velocity=(0.5, 0.0)),
        _view("r2", 2.0, velocity=(-0.5, 0.0)),
    )
    assert status is FSMStatus.AVOIDING


def test_unknown_velocity_uses_worst_case_speed():
    status, _ = _check(
        _view("r1", 1.0, velocity=None),
        _view("r2", 2.0),
        worst_case_speed_mps=1.0,
    )
    assert status is FSMStatus.AVOIDING
