from unittest.mock import MagicMock

from entities.enums import (
    FSMStatus,
    LocalizationStatus,
    PoseSource,
    TrafficPattern,
)
from entities.events import CollisionStallEvent
from entities.fleet_snapshot import FleetRobotView, FleetSnapshot
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from use_cases.event_dispatcher import EventDispatcher
from use_cases.robot_agent import RobotAgent
from use_cases.robot_registry import RobotRegistry
from use_cases.traffic_manager import TrafficManager


def _view(robot_id, coords, path=()):
    return FleetRobotView(
        robot_id=robot_id,
        pose_estimate=PoseEstimate(
            Pose(*coords), 0.03, PoseSource.FUSED, 1.0
        ),
        observed_chassis_velocity=(0.0, 0.0),
        footprint_radius_m=0.35,
        active_path=tuple(path),
        localization_status=LocalizationStatus.TRUSTED,
        active_faults=frozenset(),
        fsm_status=FSMStatus.NAVIGATING if path else FSMStatus.IDLE,
    )


def _event(blockers):
    return CollisionStallEvent(
        sim_time=1.0,
        robot_id="r1",
        coords=(2.0, 2.0),
        blocking_robot_ids=list(blockers),
    )


def _snapshot(*views):
    return FleetSnapshot(version=1, sim_time=1.0, robots=tuple(views))


def _manager():
    topology = MagicMock()
    topology.grid_width = 10
    topology.grid_height = 10
    return TrafficManager(topology, MagicMock(), RobotRegistry())


def test_traffic_manager_categorize_head_on():
    snapshot = _snapshot(
        _view("r1", (2.0, 2.0), ((3.0, 2.0),)),
        _view("r2", (3.0, 2.0), ((2.0, 2.0),)),
    )

    assert _manager()._categorize_pattern(
        _event(["r2"]), snapshot
    ) is TrafficPattern.HEAD_ON


def test_traffic_manager_categorize_rear_end():
    snapshot = _snapshot(
        _view("r1", (2.0, 2.0), ((3.0, 2.0),)),
        _view("r2", (3.0, 2.0), ((4.0, 2.0),)),
    )

    assert _manager()._categorize_pattern(
        _event(["r2"]), snapshot
    ) is TrafficPattern.REAR_END


def test_traffic_manager_categorize_crossing():
    snapshot = _snapshot(
        _view("r1", (2.0, 2.0), ((3.0, 2.0),)),
        _view("r2", (2.5, 1.5), ((2.5, 2.5),)),
    )

    assert _manager()._categorize_pattern(
        _event(["r2"]), snapshot
    ) is TrafficPattern.CROSSING


def test_traffic_manager_categorize_stuck_between():
    snapshot = _snapshot(
        _view("r1", (2.0, 2.0), ((3.0, 2.0),)),
        _view("r2", (3.0, 2.0)),
        _view("r3", (1.0, 2.0)),
    )

    assert _manager()._categorize_pattern(
        _event(["r2", "r3"]), snapshot
    ) is TrafficPattern.STUCK_BETWEEN


def test_traffic_manager_categorize_parked_as_rear_end():
    snapshot = _snapshot(
        _view("r1", (2.0, 2.0), ((3.0, 2.0),)),
        _view("r2", (3.0, 2.0)),
    )

    assert _manager()._categorize_pattern(
        _event(["r2"]), snapshot
    ) is TrafficPattern.REAR_END


def test_traffic_manager_event_dispatcher_integration():
    manager = _manager()
    dispatcher = EventDispatcher()
    dispatcher.subscribe(CollisionStallEvent, manager.handle_stall_event)
    strategy = MagicMock()
    manager.set_strategy(TrafficPattern.HEAD_ON, strategy)

    agent1 = MagicMock(spec=RobotAgent)
    agent1.path = [(3.0, 2.0)]
    agent1.path_index = 0
    agent1.current_task = MagicMock()
    agent2 = MagicMock(spec=RobotAgent)
    agent2.path = [(2.0, 2.0)]
    agent2.path_index = 0
    agents = {"r1": agent1, "r2": agent2}
    manager.robots.replace(agents.items())
    snapshot = _snapshot(
        _view("r1", (2.0, 2.0), ((3.0, 2.0),)),
        _view("r2", (3.0, 2.0), ((2.0, 2.0),)),
    )
    manager.update_snapshot(snapshot)

    dispatcher.dispatch(_event(["r2"]))

    strategy.resolve.assert_called_once()
    assert strategy.resolve.call_args.args[2] is snapshot
    blocked_cells = strategy.resolve.call_args.args[3]
    assert isinstance(blocked_cells, frozenset)
    assert (3, 2) in blocked_cells
