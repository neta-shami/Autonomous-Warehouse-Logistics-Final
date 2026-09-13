import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from entities.drive_command import DriveCommand
from entities.enums import FSMStatus, LocalizationStatus
from entities.events import CollisionStallEvent, RobotIdleEvent
from entities.fleet_snapshot import FleetSnapshot
from entities.pose import Pose
from entities.runtime_control import TickContext
from use_cases.robot_states import (
    AvoidingState,
    IdleState,
    ManipulatingState,
    NavigatingState,
)


class TestRobotStates(unittest.TestCase):
    def setUp(self):
        self.agent = MagicMock()
        self.agent.state = MagicMock()
        self.agent.state.robot_id = "r1"
        self.agent.state.current_coords = (0.0, 0.0)
        self.agent.state.pose_estimate = SimpleNamespace(pose=Pose(0.0, 0.0))
        self.agent.state.localization_status = LocalizationStatus.TRUSTED
        self.agent.local_avoidance = MagicMock()
        self.agent.event_bus = []
        self.agent.local_avoidance.check_collisions.return_value = (None, [])
        self.agent.path = [(1, 1)]
        self.agent.path_index = 0
        self.agent.WAYPOINT_TOLERANCE = 0.15
        self.agent.FINAL_WAYPOINT_TOLERANCE = 0.08
        self.context = TickContext(
            frame=None,
            fleet_snapshot=FleetSnapshot(1, 2.5, ()),
            sim_time=2.5,
            dt=0.02,
            gate_reasons=frozenset(),
        )

    def test_idle_state_publishes_idle_once(self):
        state = IdleState()

        state.update(self.agent, self.context)
        state.update(self.agent, self.context)

        self.assertEqual(state.fsm_status, FSMStatus.IDLE)
        self.agent.apply_drive.assert_not_called()
        self.assertEqual(len(self.agent.event_bus), 1)
        self.assertIsInstance(self.agent.event_bus[0], RobotIdleEvent)

    def test_navigating_state_applies_waypoint_command(self):
        state = NavigatingState()
        command = DriveCommand(0.4, 0.2)
        self.agent.waypoint_controller.command_for.return_value = command

        state.update(self.agent, self.context)

        self.assertEqual(state.fsm_status, FSMStatus.NAVIGATING)
        self.agent.waypoint_controller.command_for.assert_called_once_with(
            self.agent.state.pose_estimate.pose,
            (1, 1),
            self.agent.FINAL_WAYPOINT_TOLERANCE,
        )
        self.agent.apply_drive.assert_called_once_with(command)

    def test_navigating_state_stops_for_path_with_wrong_leg_goal(self):
        state = NavigatingState()
        self.agent.current_leg_goal = (2, 2)

        state.update(self.agent, self.context)

        self.agent.stop_drive.assert_called_once_with()
        self.agent.apply_drive.assert_not_called()

    def test_navigating_state_collision_stops_and_avoids(self):
        state = NavigatingState()
        self.agent.local_avoidance.check_collisions.return_value = (
            FSMStatus.AVOIDING,
            ["r2"],
        )

        state.update(self.agent, self.context)

        self.agent.stop_drive.assert_called_once_with()
        self.agent.transition_to.assert_called_once()
        next_state = self.agent.transition_to.call_args.args[0]
        self.assertIsInstance(next_state, AvoidingState)
        self.assertEqual(len(self.agent.event_bus), 1)
        self.assertIsInstance(self.agent.event_bus[0], CollisionStallEvent)
        self.assertEqual(self.agent.event_bus[0].sim_time, 2.5)

    def test_navigating_state_arrival_stops_before_manipulation(self):
        state = NavigatingState()
        self.agent.waypoint_controller.command_for.return_value = DriveCommand(
            0.0, 0.0
        )

        state.update(self.agent, self.context)

        self.agent.stop_drive.assert_called_once_with()
        self.assertEqual(self.agent.path_index, 1)
        next_state = self.agent.transition_to.call_args.args[0]
        self.assertIsInstance(next_state, ManipulatingState)

    def test_avoiding_state_waits_if_blocked(self):
        state = AvoidingState()
        self.agent.local_avoidance.check_collisions.return_value = (
            FSMStatus.AVOIDING,
            ["r2"],
        )

        state.update(self.agent, self.context)

        self.agent.stop_drive.assert_called_once_with()
        self.agent.transition_to.assert_not_called()

    def test_avoiding_state_transitions_to_navigating_if_clear(self):
        state = AvoidingState()

        state.update(self.agent, self.context)

        self.agent.stop_drive.assert_called_once_with()
        next_state = self.agent.transition_to.call_args.args[0]
        self.assertIsInstance(next_state, NavigatingState)

    def test_manipulating_state_executes_strategy(self):
        state = ManipulatingState()
        strategy = MagicMock()
        strategy.execute.return_value = True
        self.agent.task_strategy = strategy
        self.agent.current_task = MagicMock()

        state.update(self.agent, self.context)

        self.assertEqual(state.fsm_status, FSMStatus.MANIPULATING)
        self.agent.stop_drive.assert_called_once_with()
        strategy.execute.assert_called_once_with(self.agent, 2.5)
        next_state = self.agent.transition_to.call_args.args[0]
        self.assertIsInstance(next_state, IdleState)
        self.assertIsNone(self.agent.current_task)
        self.assertFalse(self.agent.state.is_busy)


if __name__ == "__main__":
    unittest.main()
