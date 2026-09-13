"""Task strategies with explicit, recoverable execution phases."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, FrozenSet, Tuple

from entities.enums import (
    FaultCode,
    LocalizationStatus,
    ManipulatorAbortResult,
    ManipulatorResult,
    TaskExecutionPhase,
    TaskStatus,
)
from entities.events import (
    PackageDroppedEvent,
    PackagePickedEvent,
    TaskCompletedEvent,
)
from entities.pose import Pose
from use_cases.dynamic_obstacles import blocked_cells_for_robot

if TYPE_CHECKING:
    from use_cases.robot_agent import RobotAgent


class ITaskStrategy(ABC):
    @abstractmethod
    def execute(self, agent: "RobotAgent", sim_time: float) -> bool:
        """Advance the public task phase; return true only when terminal."""
        raise NotImplementedError


class TransferTaskStrategy(ITaskStrategy):
    """Execute guarded pick/drop operations without private phase authority."""

    def __init__(self):
        self._hardware_completed_phase = None
        self._operation_started_phase = None

    def execute(self, agent: "RobotAgent", sim_time: float) -> bool:
        task = agent.current_task
        if task is None:
            return True
        if not self._safe_to_manipulate(agent):
            return False

        if task.phase is TaskExecutionPhase.PICKING:
            return self._pick(agent, sim_time)
        if task.phase is TaskExecutionPhase.DROPPING:
            return self._drop(agent, sim_time)
        return False

    def _pick(self, agent: "RobotAgent", sim_time: float) -> bool:
        task = agent.current_task
        if (
            task is None
            or task.source_coords is None
            or task.source_tier is None
            or task.payload_id is None
        ):
            raise RuntimeError("pick requires a complete transfer task")
        if self._hardware_completed_phase is not TaskExecutionPhase.PICKING:
            if self._operation_started_phase is not TaskExecutionPhase.PICKING:
                if agent.state.payload_present is not False:
                    return False
                self._operation_started_phase = TaskExecutionPhase.PICKING
            approach = agent.topology.get_approach_direction(
                task.source_coords, self._current_grid(agent)
            )
            result = agent.manipulator_system.pick(
                agent.state.robot_id,
                task.payload_id,
                task.source_tier,
                approach,
            )
            if result is ManipulatorResult.FAILED:
                return self._fail_task(agent, sim_time)
            if result is not ManipulatorResult.SUCCEEDED:
                return False
            self._hardware_completed_phase = TaskExecutionPhase.PICKING
            self._operation_started_phase = None

        # The arm result alone is insufficient: wait for committed debounce.
        if agent.state.payload_present is not True:
            return False
        agent.event_bus.append(
            PackagePickedEvent(
                sim_time, agent.state.robot_id, task, task.lease_generation
            )
        )
        task.phase = TaskExecutionPhase.TO_TARGET
        fallback_access = agent.topology.get_access_point(
            task.target_coords, self._current_grid(agent)
        )
        snapshot = agent.latest_fleet_snapshot
        blocked_cells: FrozenSet[Tuple[int, int]]
        if snapshot is not None:
            active_paths = tuple(
                robot.active_path
                for robot in snapshot.robots
                if robot.robot_id != agent.state.robot_id
            )
            blocked_cells = blocked_cells_for_robot(
                snapshot,
                agent.topology,
                agent.state.robot_id,
                agent.FOOTPRINT_RADIUS_M,
            )
        else:
            active_paths = ()
            blocked_cells = frozenset()
        estimate = agent.state.pose_estimate
        if estimate is None:
            return False
        target, target_path = agent.path_planner.find_path_to_access(
            estimate.pose,
            task.target_coords,
            agent.topology,
            active_paths,
            blocked_cells=blocked_cells,
            footprint_radius_m=agent.FOOTPRINT_RADIUS_M,
        )
        if target is not None and target_path:
            agent.current_leg_goal = target
            agent.set_path(target_path)
        else:
            agent.current_leg_goal = fallback_access
            agent.path = []
            agent.path_index = 0
            agent.route_replan_required = True
        self._hardware_completed_phase = None
        from use_cases.robot_states import NavigatingState

        agent.transition_to(NavigatingState())
        return False

    def _drop(self, agent: "RobotAgent", sim_time: float) -> bool:
        task = agent.current_task
        if task is None or task.payload_id is None:
            raise RuntimeError("drop requires an identified transfer payload")
        if self._hardware_completed_phase is not TaskExecutionPhase.DROPPING:
            if self._operation_started_phase is not TaskExecutionPhase.DROPPING:
                if agent.state.payload_present is not True:
                    return False
                self._operation_started_phase = TaskExecutionPhase.DROPPING
            approach = agent.topology.get_approach_direction(
                task.target_coords, self._current_grid(agent)
            )
            result = agent.manipulator_system.drop(
                agent.state.robot_id,
                task.payload_id,
                task.target_tier,
                approach,
            )
            if result is ManipulatorResult.FAILED:
                return self._fail_task(agent, sim_time)
            if result is not ManipulatorResult.SUCCEEDED:
                return False
            self._hardware_completed_phase = TaskExecutionPhase.DROPPING
            self._operation_started_phase = None

        if agent.state.payload_present is not False:
            return False
        agent.event_bus.append(
            PackageDroppedEvent(
                sim_time, agent.state.robot_id, task, task.lease_generation
            )
        )
        task.phase = TaskExecutionPhase.COMPLETE
        task.update_status(TaskStatus.COMPLETED)
        agent.event_bus.append(
            TaskCompletedEvent(
                sim_time=sim_time,
                task_id=task.task_id,
                robot_id=agent.state.robot_id,
                success=True,
                task=task,
                lease_generation=task.lease_generation,
            )
        )
        return True

    @staticmethod
    def _safe_to_manipulate(agent: "RobotAgent") -> bool:
        estimate = agent.state.pose_estimate
        goal = agent.current_leg_goal
        task = agent.current_task
        return (
            agent.state.localization_status is LocalizationStatus.TRUSTED
            and estimate is not None
            and goal is not None
            and estimate.pose.distance_to(Pose(*goal))
            <= agent.FINAL_WAYPOINT_TOLERANCE
            and task is not None
            and task.payload_id == agent.state.expected_payload_id
        )

    @staticmethod
    def _current_grid(agent: "RobotAgent") -> Tuple[int, int]:
        return (
            int(round(agent.state.current_coords[0])),
            int(round(agent.state.current_coords[1])),
        )

    @staticmethod
    def _fail_task(agent: "RobotAgent", sim_time: float) -> bool:
        task = agent.current_task
        if task is None:
            raise RuntimeError("cannot fail a robot without an active task")
        abort_result = agent.manipulator_system.abort(agent.state.robot_id)
        payload_may_be_in_transit = (
            task.phase in (
                TaskExecutionPhase.TO_TARGET,
                TaskExecutionPhase.DROPPING,
            )
            or agent.state.payload_present is True
        )
        if (
            abort_result is ManipulatorAbortResult.RECONCILIATION_REQUIRED
            or payload_may_be_in_transit
        ):
            task.update_status(TaskStatus.RECOVERY_REQUIRED)
            agent.state.activate_faults(FaultCode.MANIPULATOR_FAILURE)
            from use_cases.robot_states import FaultedState

            agent.transition_to(FaultedState())
            return False
        task.update_status(TaskStatus.FAILED)
        agent.event_bus.append(
            TaskCompletedEvent(
                sim_time=sim_time,
                task_id=task.task_id,
                robot_id=agent.state.robot_id,
                success=False,
                task=task,
                lease_generation=task.lease_generation,
            )
        )
        return True


class ParkTaskStrategy(ITaskStrategy):
    def execute(self, agent: "RobotAgent", sim_time: float) -> bool:
        task = agent.current_task
        if task:
            task.phase = TaskExecutionPhase.COMPLETE
            task.update_status(TaskStatus.COMPLETED)
            agent.event_bus.append(
                TaskCompletedEvent(
                    sim_time=sim_time,
                    task_id=task.task_id,
                    robot_id=agent.state.robot_id,
                    success=True,
                    task=task,
                    lease_generation=task.lease_generation,
                )
            )
        return True
