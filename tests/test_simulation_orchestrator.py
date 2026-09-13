import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from adapters.simulation_orchestrator import SimulationOrchestrator
from entities.enums import (
    CommandGateReason,
    FaultCode,
    FSMStatus,
    LocalizationStatus,
)
from entities.events import RobotIdleEvent, TaskCompletedEvent
from entities.task import Task
from interfaces.sensor_read_error import SensorReadError
from use_cases.robot_registry import RobotRegistry
from use_cases.robot_states import FaultedState


def _unlocalized_agent():
    """Return a valid command-free agent double for orchestration tests."""
    agent = MagicMock()
    agent.state = SimpleNamespace(
        pose_estimate=None,
        observed_chassis_velocity=None,
        localization_status=LocalizationStatus.UNINITIALIZED,
        active_faults=frozenset(),
    )
    agent.fsm_status = FSMStatus.RELOCALIZING
    agent.FOOTPRINT_RADIUS_M = 0.35
    agent.path = []
    agent.path_index = 0
    agent.event_bus = []
    agent.assess.return_value.stop_required = False
    agent.assess.return_value.observation.sequence = None
    return agent

class TestSimulationOrchestrator(unittest.TestCase):
    def setUp(self):
        self.mock_env = MagicMock()
        self.mock_scenario = MagicMock()
        self.mock_sensor = MagicMock()
        self.mock_controller = MagicMock()
        self.mock_manipulator = MagicMock()
        self.mock_topology = MagicMock()
        self.mock_fleet = MagicMock()
        self.mock_traffic = MagicMock()
        self.mock_metrics = MagicMock()
        self.mock_path = MagicMock()
        self.robot_registry = RobotRegistry()
        self.mock_env.get_time.return_value = 2.5

        self.orchestrator = SimulationOrchestrator(
            environment=self.mock_env,
            scenario_manager=self.mock_scenario,
            sensor_adapter=self.mock_sensor,
            controller=self.mock_controller,
            attachment_synchronizer=self.mock_controller,
            manipulator=self.mock_manipulator,
            topology=self.mock_topology,
            fleet_manager=self.mock_fleet,
            traffic_manager=self.mock_traffic,
            metrics=self.mock_metrics,
            path_planner=self.mock_path,
            robot_registry=self.robot_registry,
        )
        self.orchestrator.physics_steps_per_logic_tick = 2

    def test_step_delegates_to_environment_and_scenario(self):
        # Action
        self.orchestrator.step()

        # Assert: physics is decimated, logic runs once per tick
        self.assertEqual(self.mock_env.step.call_count, 2)
        self.mock_scenario.tick.assert_called_once_with(2.5)

    def test_step_syncs_gripped_packages_on_every_physics_step(self):
        """A carried package must not lag behind its carrier between ticks."""
        self.orchestrator.step()
        self.assertEqual(self.mock_controller.sync_attachments.call_count, 2)

    def test_step_accumulates_transient_collisions_across_physics_steps(self):
        self.orchestrator.agents = {
            "r1": _unlocalized_agent(),
            "r2": _unlocalized_agent(),
        }
        collision_pairs = {("r1", "r2"), ("r1", "wall_west")}
        self.mock_env.get_robot_collisions.side_effect = [
            {("r1", "wall_west")},
            {("r1", "r2")},
        ]

        self.orchestrator.step()

        self.assertEqual(self.mock_env.get_robot_collisions.call_count, 2)
        self.mock_env.get_robot_collisions.assert_called_with(("r1", "r2"))
        self.mock_metrics.record_robot_collisions.assert_called_once_with(
            collision_pairs
        )

    def test_tick_orders_all_reads_assessments_and_commits_before_actions(self):
        order = []
        r1 = _unlocalized_agent()
        r2 = _unlocalized_agent()
        # Reverse insertion order to prove registration order cannot affect acts.
        self.orchestrator.agents = {"r2": r2, "r1": r1}
        self.mock_sensor.read_frame.side_effect = (
            lambda robot_id: order.append(f"read:{robot_id}") or MagicMock()
        )
        for robot_id, agent in (("r1", r1), ("r2", r2)):
            agent.assess.side_effect = (
                lambda *_args, rid=robot_id, result=agent.assess.return_value:
                order.append(f"assess:{rid}") or result
            )
            agent.commit.side_effect = (
                lambda *_args, rid=robot_id: order.append(f"commit:{rid}")
            )
            agent.act.side_effect = (
                lambda *_args, rid=robot_id: order.append(f"act:{rid}")
            )

        self.orchestrator.step()

        self.assertEqual(
            order,
            [
                "read:r1", "read:r2",
                "assess:r1", "assess:r2",
                "commit:r1", "commit:r2",
                "act:r1", "act:r2",
            ],
        )
        r1_context = r1.act.call_args.args[0]
        r2_context = r2.act.call_args.args[0]
        self.assertIs(r1_context.fleet_snapshot, r2_context.fleet_snapshot)
        self.assertEqual(r1_context.fleet_snapshot.version, 1)
        self.assertEqual(r1_context.fleet_snapshot.sim_time, 2.5)

    def test_sensor_read_error_gates_that_robot_in_the_same_tick(self):
        r1 = _unlocalized_agent()
        r2 = _unlocalized_agent()
        self.orchestrator.agents = {"r1": r1, "r2": r2}
        self.mock_sensor.read_frame.side_effect = [
            SensorReadError("r1 failed"),
            MagicMock(),
        ]
        r1.assess.return_value.stop_required = True

        self.orchestrator.step()

        r1.assess.assert_called_once_with(None, 2.5, 0.0)
        r1_context = r1.act.call_args.args[0]
        self.assertIn(CommandGateReason.LOCAL_SAFETY, r1_context.gate_reasons)
        self.assertNotIn(
            CommandGateReason.LOCAL_SAFETY,
            r2.act.call_args.args[0].gate_reasons,
        )

    def test_events_dispatch_only_after_every_robot_has_acted(self):
        order = []
        r1 = _unlocalized_agent()
        r2 = _unlocalized_agent()
        r1.event_bus.append(
            RobotIdleEvent(2.5, robot_id="r1", current_coords=(1.0, 1.0))
        )
        r1.act.side_effect = lambda *_args: order.append("act:r1")
        r2.act.side_effect = lambda *_args: order.append("act:r2")
        self.mock_fleet.handle_robot_idle.side_effect = (
            lambda _event: order.append("event")
        )
        self.orchestrator.agents = {"r1": r1, "r2": r2}

        self.orchestrator.step()

        self.assertEqual(order, ["act:r1", "act:r2", "event"])

    def test_events_appended_by_a_handler_are_drained_in_the_same_tick(self):
        """Nested agent events must not be erased by the dispatch cleanup."""
        r1 = _unlocalized_agent()
        r1.event_bus.append(
            RobotIdleEvent(2.5, robot_id="r1", current_coords=(1.0, 1.0))
        )
        task = Task.park("park-r1", (1, 9))
        completed = TaskCompletedEvent(2.5, task.task_id, "r1", True, task)
        self.mock_fleet.handle_robot_idle.side_effect = (
            lambda _event: r1.event_bus.append(completed)
        )
        self.orchestrator.agents = {"r1": r1}

        self.orchestrator.step()

        self.mock_metrics.handle_task_completed.assert_called_once_with(completed)
        self.assertEqual(r1.event_bus, [])

    def test_default_logic_rate_is_fifty_hertz(self):
        """0.002 s physics timestep times 10 steps gives a 50 Hz logic tick."""
        self.assertEqual(self.orchestrator.PHYSICS_STEPS_PER_LOGIC_TICK, 10)

    def test_rejects_a_decimation_factor_below_one(self):
        """Zero would freeze physics; negatives are meaningless."""
        for invalid in (0, -1):
            with self.assertRaises(ValueError):
                SimulationOrchestrator(
                    environment=self.mock_env,
                    scenario_manager=self.mock_scenario,
                    sensor_adapter=self.mock_sensor,
                    controller=self.mock_controller,
                    attachment_synchronizer=self.mock_controller,
                    manipulator=self.mock_manipulator,
                    topology=self.mock_topology,
                    fleet_manager=self.mock_fleet,
                    traffic_manager=self.mock_traffic,
                    metrics=self.mock_metrics,
                    path_planner=self.mock_path,
                    robot_registry=RobotRegistry(),
                    physics_steps_per_logic_tick=invalid
                )

    def test_spawn_package_delegates_to_scenario(self):
        self.mock_scenario.spawn_package.return_value = True
        result = self.orchestrator.spawn_package("pkg1", (2, 2), 1)
        self.assertTrue(result)
        self.mock_scenario.spawn_package.assert_called_once_with(
            "pkg1", (2, 2), 1, 2.5
        )

    def test_spawn_package_rejects_an_invalid_shelf_before_mutating_world(self):
        self.mock_topology.validate_shelf_slot.side_effect = ValueError(
            "not a shelf"
        )

        with self.assertRaisesRegex(ValueError, "not a shelf"):
            self.orchestrator.spawn_package("pkg1", (3, 3), 1)

    def test_reconciliation_request_uses_the_audited_fault_reset_flow(self):
        self.orchestrator.register_robot("r1", (1.0, 9.0))
        agent = self.orchestrator.agents["r1"]
        agent.state.activate_faults(FaultCode.MANIPULATOR_FAILURE)
        agent.transition_to(FaultedState(), 1.0)

        accepted = self.orchestrator.request_fault_reset(
            "r1",
            FaultCode.MANIPULATOR_FAILURE,
            inventory_reconciled=True,
        )

        self.assertTrue(accepted)
        self.assertFalse(agent.state.active_faults)
        self.assertEqual(agent.event_bus[-1].cleared_fault,
                         FaultCode.MANIPULATOR_FAILURE)

        self.mock_scenario.spawn_package.assert_not_called()

if __name__ == '__main__':
    unittest.main()
