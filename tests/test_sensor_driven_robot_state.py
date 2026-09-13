"""Belief-state and atomic runtime value contract tests."""

from dataclasses import FrozenInstanceError

import pytest

from entities.enums import (
    FaultCode,
    FSMStatus,
    LocalizationStatus,
    PoseSource,
    ZoneType,
)
from entities.fleet_snapshot import FleetRobotView, FleetSnapshot
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from entities.robot_state import RobotState
from entities.warehouse_topology import WarehouseTopology
from use_cases.robot_health_monitor import RobotHealthMonitor


def _estimate(x=1.25, y=2.5, sim_time=1.0):
    return PoseEstimate(Pose(x, y), 0.03, PoseSource.FUSED, sim_time)


def test_new_robot_has_no_claimed_pose_until_localization_commits():
    state = RobotState(
        robot_id="r1",
        pose_estimate=None,
        observed_chassis_velocity=None,
        expected_payload_id=None,
        payload_present=None,
        is_busy=False,
        home_base_coords=(1, 9),
    )

    assert state.localization_status is LocalizationStatus.UNINITIALIZED
    with pytest.raises(RuntimeError, match="no usable pose"):
        _ = state.current_coords


def test_localization_and_observation_commit_to_belief_state():
    state = RobotState("r1", None, None, None, None, False, (1, 9))
    estimate = _estimate()

    state.apply_localization(estimate, LocalizationStatus.TRUSTED)
    state.apply_observation((0.4, -0.2), True)

    assert state.pose_estimate is estimate
    assert state.current_coords == (1.25, 2.5)
    assert state.observed_chassis_velocity == (0.4, -0.2)
    assert state.payload_present is True


@pytest.mark.parametrize(
    ("estimate", "status"),
    [
        (None, LocalizationStatus.TRUSTED),
        (_estimate(), LocalizationStatus.UNINITIALIZED),
        (_estimate(), LocalizationStatus.LOST),
    ],
)
def test_rejects_incoherent_localization_state(estimate, status):
    state = RobotState("r1", None, None, None, None, False, (1, 9))

    with pytest.raises(ValueError, match="localization"):
        state.apply_localization(estimate, status)


def test_fleet_snapshot_is_immutable_and_contains_one_bounded_view():
    estimate = _estimate()
    view = FleetRobotView(
        robot_id="r1",
        pose_estimate=estimate,
        observed_chassis_velocity=(0.0, 0.0),
        footprint_radius_m=0.35,
        active_path=((2, 9),),
        localization_status=LocalizationStatus.TRUSTED,
        active_faults=frozenset({FaultCode.DRIVE_STALL}),
        fsm_status=FSMStatus.NAVIGATING,
    )
    snapshot = FleetSnapshot(version=3, sim_time=1.5, robots=(view,))

    assert snapshot.robot("r1") is view
    with pytest.raises(FrozenInstanceError):
        snapshot.version = 4


def test_pose_feasibility_is_separate_from_localization():
    topology = WarehouseTopology(
        10,
        10,
        0.0,
        10.0,
        0.0,
        10.0,
        {(2, 2): ZoneType.SHELF},
    )

    assert topology.is_pose_feasible(Pose(1.0, 1.0), 0.35)
    assert not topology.is_pose_feasible(Pose(0.2, 1.0), 0.35)
    assert not topology.is_pose_feasible(Pose(2.0, 2.0), 0.35)


def test_empty_health_monitor_has_explicitly_no_findings():
    assert RobotHealthMonitor([]).evaluate(object()) == ()
