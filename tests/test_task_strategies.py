import unittest
from unittest.mock import MagicMock

from entities.enums import (
    CardinalDirection,
    FSMStatus,
    LocalizationStatus,
    ManipulatorAbortResult,
    ManipulatorResult,
    PoseSource,
    TaskExecutionPhase,
    TaskStatus,
)
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from use_cases.task_strategies import ParkTaskStrategy, TransferTaskStrategy


class TestTaskStrategies(unittest.TestCase):
    def setUp(self):
        self.agent = MagicMock()
        self.agent.state = MagicMock()
        self.agent.state.robot_id = "r1"
        self.agent.state.current_coords = (0, 0)
        self.agent.state.pose_estimate = PoseEstimate(
            Pose(0, 0), 0.03, PoseSource.FUSED, 1.0
        )
        self.agent.state.localization_status = LocalizationStatus.TRUSTED
        self.agent.state.payload_present = False
        self.agent.state.expected_payload_id = "p1"
        self.agent.current_leg_goal = (0, 0)
        self.agent.FINAL_WAYPOINT_TOLERANCE = 0.1
        self.agent.event_bus = []
        self.agent.manipulator_system = MagicMock()
        self.agent.path_planner = MagicMock()
        self.agent.topology = MagicMock()

        self.task = MagicMock()
        self.task.task_id = "t1"
        self.task.source_tier = 1
        self.task.source_coords = (1, 0)
        self.task.target_tier = 0
        self.task.target_coords = (5, 5)
        self.task.payload_id = "p1"
        self.task.lease_generation = 1
        self.task.phase = TaskExecutionPhase.PICKING
        self.agent.current_task = self.task

    def test_park_task_strategy(self):
        strategy = ParkTaskStrategy()

        result = strategy.execute(self.agent, 3.5)

        self.assertTrue(result)
        self.task.update_status.assert_called_with(TaskStatus.COMPLETED)
        self.assertEqual(len(self.agent.event_bus), 1)
        self.assertEqual(self.agent.event_bus[0].task_id, "t1")
        self.assertEqual(self.agent.event_bus[0].sim_time, 3.5)

    def test_transfer_task_strategy_pick_phase(self):
        strategy = TransferTaskStrategy()

        self.agent.manipulator_system.pick.return_value = ManipulatorResult.SUCCEEDED
        self.agent.path_planner.find_path_to_access.return_value = (
            (4, 5),
            [(1, 1), (2, 2)],
        )
        self.agent.topology.get_zone_type.return_value = None
        self.agent.topology.get_approach_direction.return_value = CardinalDirection.POS_X

        result = strategy.execute(self.agent, 3.5)
        self.assertFalse(result)
        self.assertEqual(self.agent.event_bus, [])
        self.agent.state.payload_present = True
        result = strategy.execute(self.agent, 3.6)

        self.assertFalse(result)
        self.agent.manipulator_system.pick.assert_called_with(
            "r1", "p1", 1, CardinalDirection.POS_X
        )
        self.agent.topology.get_approach_direction.assert_any_call((1, 0), (0, 0))

        self.assertEqual(len(self.agent.event_bus), 1)
        self.assertEqual(self.agent.event_bus[0].__class__.__name__, "PackagePickedEvent")
        self.assertEqual(self.agent.event_bus[0].sim_time, 3.6)

        # Should transition to Navigating
        self.agent.transition_to.assert_called_once()
        args, _ = self.agent.transition_to.call_args
        self.assertEqual(args[0].__class__.__name__, "NavigatingState")

        self.agent.set_path.assert_called_once_with([(1, 1), (2, 2)])
        self.assertEqual(self.agent.current_leg_goal, (4, 5))
        self.assertIs(self.task.phase, TaskExecutionPhase.TO_TARGET)

    def test_transfer_task_strategy_drop_phase(self):
        strategy = TransferTaskStrategy()
        self.task.phase = TaskExecutionPhase.DROPPING
        self.agent.state.payload_present = True

        self.agent.manipulator_system.drop.return_value = ManipulatorResult.SUCCEEDED
        self.agent.topology.get_approach_direction.return_value = CardinalDirection.NEG_Y

        result = strategy.execute(self.agent, 3.5)
        self.assertEqual(self.agent.event_bus, [])
        self.agent.state.payload_present = False
        result = strategy.execute(self.agent, 3.6)

        self.assertTrue(result)
        self.agent.manipulator_system.drop.assert_called_with(
            "r1", "p1", 0, CardinalDirection.NEG_Y
        )
        self.agent.topology.get_approach_direction.assert_called_with((5, 5), (0, 0))

        self.assertEqual(len(self.agent.event_bus), 2)
        self.assertEqual(self.agent.event_bus[0].__class__.__name__, "PackageDroppedEvent")
        self.assertEqual(self.agent.event_bus[1].__class__.__name__, "TaskCompletedEvent")
        self.assertTrue(all(event.sim_time == 3.6 for event in self.agent.event_bus))

        self.task.update_status.assert_called_with(TaskStatus.COMPLETED)

    def test_in_progress_result_does_not_emit_success_events(self):
        strategy = TransferTaskStrategy()
        self.agent.manipulator_system.pick.return_value = ManipulatorResult.IN_PROGRESS
        self.agent.topology.get_approach_direction.return_value = CardinalDirection.POS_X

        result = strategy.execute(self.agent, 3.5)

        self.assertFalse(result)
        self.assertEqual(self.agent.event_bus, [])
        self.agent.transition_to.assert_not_called()

    def test_untrusted_localization_blocks_manipulator(self):
        self.agent.state.localization_status = LocalizationStatus.DEGRADED

        result = TransferTaskStrategy().execute(self.agent, 3.5)

        self.assertFalse(result)
        self.agent.manipulator_system.pick.assert_not_called()

    def test_wrong_leg_position_blocks_manipulator(self):
        self.agent.current_leg_goal = (1, 0)

        result = TransferTaskStrategy().execute(self.agent, 3.5)

        self.assertFalse(result)
        self.agent.manipulator_system.pick.assert_not_called()

    def test_payload_identity_mismatch_blocks_manipulator(self):
        self.agent.state.expected_payload_id = "another"

        result = TransferTaskStrategy().execute(self.agent, 3.5)

        self.assertFalse(result)
        self.agent.manipulator_system.pick.assert_not_called()

    def test_failed_result_fails_task_without_inventory_event(self):
        strategy = TransferTaskStrategy()
        self.agent.manipulator_system.pick.return_value = ManipulatorResult.FAILED
        self.agent.topology.get_approach_direction.return_value = CardinalDirection.POS_X

        result = strategy.execute(self.agent, 3.5)

        self.assertTrue(result)
        self.task.update_status.assert_called_with(TaskStatus.FAILED)
        self.assertEqual(len(self.agent.event_bus), 1)
        event = self.agent.event_bus[0]
        self.assertEqual(event.__class__.__name__, "TaskCompletedEvent")
        self.assertFalse(event.success)
        self.assertEqual(event.sim_time, 3.5)

    def test_failed_result_after_commit_requires_recovery_without_events(self):
        strategy = TransferTaskStrategy()
        self.agent.state.active_faults = frozenset()
        self.agent.manipulator_system.pick.return_value = ManipulatorResult.FAILED
        self.agent.manipulator_system.abort.return_value = (
            ManipulatorAbortResult.RECONCILIATION_REQUIRED
        )
        self.agent.topology.get_approach_direction.return_value = (
            CardinalDirection.POS_X
        )

        result = strategy.execute(self.agent, 3.5)

        self.assertFalse(result)
        self.task.update_status.assert_called_once_with(
            TaskStatus.RECOVERY_REQUIRED
        )
        self.assertEqual(self.agent.event_bus, [])
        next_state = self.agent.transition_to.call_args.args[0]
        self.assertEqual(next_state.fsm_status, FSMStatus.FAULTED)

    def test_failed_drop_quarantines_loaded_task_even_when_abort_is_clean(self):
        self.task.phase = TaskExecutionPhase.DROPPING
        self.agent.state.payload_present = True
        self.agent.manipulator_system.drop.return_value = ManipulatorResult.FAILED
        self.agent.manipulator_system.abort.return_value = (
            ManipulatorAbortResult.CLEAN
        )
        self.agent.topology.get_approach_direction.return_value = (
            CardinalDirection.NEG_Y
        )

        result = TransferTaskStrategy().execute(self.agent, 3.5)

        self.assertFalse(result)
        self.task.update_status.assert_called_once_with(
            TaskStatus.RECOVERY_REQUIRED
        )
        self.assertEqual(self.agent.event_bus, [])
        self.agent.state.activate_faults.assert_called_once()
        next_state = self.agent.transition_to.call_args.args[0]
        self.assertEqual(next_state.fsm_status, FSMStatus.FAULTED)

    def test_loaded_robot_waits_for_fleet_replan_when_target_has_no_path(self):
        strategy = TransferTaskStrategy()
        self.agent.manipulator_system.pick.return_value = ManipulatorResult.SUCCEEDED
        self.agent.path_planner.find_path_to_access.return_value = (None, [])
        self.agent.topology.get_approach_direction.return_value = (
            CardinalDirection.POS_X
        )
        strategy.execute(self.agent, 3.5)
        self.agent.state.payload_present = True

        result = strategy.execute(self.agent, 3.6)

        self.assertFalse(result)
        self.assertTrue(self.agent.route_replan_required)
        self.assertEqual(self.agent.path, [])
        self.assertIs(self.task.phase, TaskExecutionPhase.TO_TARGET)

if __name__ == '__main__':
    unittest.main()
