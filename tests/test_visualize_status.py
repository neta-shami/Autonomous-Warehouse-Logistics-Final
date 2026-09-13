from types import SimpleNamespace

from entities.enums import FSMStatus, PoseSource
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from visualize import format_robot_status


def _agent(pose_estimate, payload_present, status):
    return SimpleNamespace(
        state=SimpleNamespace(
            pose_estimate=pose_estimate,
            payload_present=payload_present,
        ),
        fsm_status=status,
    )


def test_status_handles_unknown_startup_beliefs():
    agent = _agent(None, None, FSMStatus.RELOCALIZING)

    assert format_robot_status("r1", agent) == (
        "  r1: RELOCALIZING @ unknown payload=unknown"
    )


def test_status_formats_localized_robot_and_frame():
    estimate = PoseEstimate(Pose(1.25, 8.75), 0.1, PoseSource.FUSED, 4.0)
    agent = _agent(estimate, True, FSMStatus.NAVIGATING)

    assert format_robot_status("r2", agent, 500) == (
        "  [500] r2: NAVIGATING @ (1.2,8.8) payload=present"
    )
