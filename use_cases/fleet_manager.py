"""Fleet task allocation, containment, and recovery routing policy."""

import uuid
from typing import FrozenSet, Optional, Set, Tuple

from entities.enums import (
    FaultCode,
    FSMStatus,
    LocalizationStatus,
    TaskExecutionPhase,
    TaskStatus,
    TaskType,
    ZoneType,
)
from entities.events import (
    FaultResetEvent,
    PayloadMismatchEvent,
    RelocalizedEvent,
    RobotFaultedEvent,
    RobotIdleEvent,
    TaskReadyEvent,
)
from entities.fleet_snapshot import FleetSnapshot
from entities.pose import Pose
from entities.task import Task
from entities.warehouse_topology import WarehouseTopology
from interfaces.i_path_planner import IPathPlanner
from interfaces.i_robot_registry import ICoordinatedRobot, IRobotRegistry
from interfaces.i_route_baseline_estimator import IRouteBaselineEstimator
from interfaces.i_task_allocation_strategy import ITaskAllocationStrategy
from use_cases.dynamic_obstacles import blocked_cells_for_robot
from use_cases.inventory_reservation_service import InventoryReservationService
from use_cases.robot_states import FaultedState, NavigatingState


class FleetManager:
    """Allocate safe work and contain disabled or newly relocated robots."""

    def __init__(
        self,
        reservation_service: InventoryReservationService,
        path_planner: IPathPlanner,
        topology: WarehouseTopology,
        allocation_strategy: ITaskAllocationStrategy,
        baseline_estimator: IRouteBaselineEstimator,
        robot_registry: IRobotRegistry,
    ) -> None:
        self.reservation_service = reservation_service
        self.path_planner = path_planner
        self.topology = topology
        self.allocation_strategy = allocation_strategy
        self.baseline_estimator = baseline_estimator
        self.robots = robot_registry
        self.latest_snapshot: Optional[FleetSnapshot] = None
        self.command_holds: FrozenSet[str] = frozenset()
        self._route_blocked_robot_ids: Set[str] = set()
        self._reset_recovery_robot_ids: Set[str] = set()

    def update_snapshot(self, snapshot: FleetSnapshot) -> None:
        """Consume one committed view and contain every newly unsafe route."""
        if not isinstance(snapshot, FleetSnapshot):
            raise TypeError("snapshot must be FleetSnapshot")
        self.latest_snapshot = snapshot
        holds = set()
        retry_ids = set(self._route_blocked_robot_ids)
        retry_ids.update(
            robot_id
            for robot_id, agent in self.robots.items()
            if agent.route_replan_required
        )
        self._route_blocked_robot_ids.clear()

        for view in snapshot.robots:
            agent = self.robots.get(view.robot_id)
            if (
                agent is None
                or view.pose_estimate is None
                or agent.current_leg_goal is None
                or agent.current_task is None
                or view.active_faults
            ):
                continue
            blocked = self._blocked_cells(
                view.robot_id, view.footprint_radius_m
            )
            route_intersects = bool(
                set(view.active_path).intersection(blocked)
            )
            if view.robot_id not in retry_ids and not route_intersects:
                continue
            holds.add(view.robot_id)
            if self._replan_agent(agent, view.pose_estimate.pose, blocked):
                if view.robot_id in self._reset_recovery_robot_ids:
                    agent.current_task.update_status(TaskStatus.IN_PROGRESS)
                    agent.transition_to(NavigatingState(), snapshot.sim_time)
                    self._reset_recovery_robot_ids.discard(view.robot_id)
            else:
                agent.route_replan_required = True
                self._route_blocked_robot_ids.add(view.robot_id)
        self.command_holds = frozenset(holds)

    def handle_task_ready(self, event: TaskReadyEvent) -> None:
        """Dispatch newly admitted work without waiting for a polling cycle."""
        self._dispatch_ready_tasks()

    def handle_relocalized(self, event: RelocalizedEvent) -> None:
        """Resume the preserved leg from the confirmed continuous pose."""
        agent = self.robots.get(event.robot_id)
        if agent is None or agent.current_task is None:
            return
        if agent.current_task.phase in (
            TaskExecutionPhase.PICKING,
            TaskExecutionPhase.DROPPING,
        ):
            agent.current_task.update_status(TaskStatus.RECOVERY_REQUIRED)
            agent.state.activate_faults(FaultCode.MANIPULATOR_FAILURE)
            agent.transition_to(FaultedState(), event.sim_time)
            return
        if agent.current_leg_goal is None or self.latest_snapshot is None:
            agent.current_task.update_status(TaskStatus.RECOVERY_REQUIRED)
            return
        view = self.latest_snapshot.robot(event.robot_id)
        blocked = self._blocked_cells(
            event.robot_id, view.footprint_radius_m
        )
        if self._replan_agent(agent, event.recovered, blocked):
            agent.transition_to(NavigatingState(), event.sim_time)
        else:
            self._route_blocked_robot_ids.add(event.robot_id)

    def handle_robot_faulted(self, event: RobotFaultedEvent) -> None:
        """Requeue only pre-pick work; quarantine loaded or interrupted work."""
        agent = self.robots.get(event.robot_id)
        if agent is None or agent.current_task is None:
            return
        task = agent.current_task
        if (
            task.phase is TaskExecutionPhase.TO_SOURCE
            and agent.state.payload_present is not True
        ):
            task.update_status(TaskStatus.QUEUED)
            agent.release_task_for_requeue(task)
            if task not in self.reservation_service.ready_queue:
                self.reservation_service.ready_queue.append(task)
            self._dispatch_ready_tasks()
            return
        task.update_status(TaskStatus.RECOVERY_REQUIRED)

    def handle_payload_mismatch(self, event: PayloadMismatchEvent) -> None:
        """Quarantine work until inventory reconciliation approves reset."""
        agent = self.robots.get(event.robot_id)
        if agent is not None and agent.current_task is not None:
            agent.current_task.update_status(TaskStatus.RECOVERY_REQUIRED)

    def handle_fault_reset(self, event: FaultResetEvent) -> None:
        """Resume only navigation-safe work after its final fault is cleared."""
        agent = self.robots.get(event.robot_id)
        if (
            agent is None
            or agent.state.active_faults
            or agent.current_task is None
        ):
            return
        task = agent.current_task
        navigation_faults = frozenset(
            {
                FaultCode.DRIVE_STALL,
                FaultCode.STATE_TIMEOUT,
                FaultCode.POSITION_PHYSICALLY_INVALID,
            }
        )
        inventory_faults = frozenset(
            {FaultCode.PAYLOAD_MISMATCH, FaultCode.MANIPULATOR_FAILURE}
        )
        navigational_phases = frozenset(
            {TaskExecutionPhase.TO_SOURCE, TaskExecutionPhase.TO_TARGET}
        )
        if event.cleared_fault in inventory_faults:
            if self._resume_reconciled_transfer(agent, event.sim_time):
                return
            self._restore_unresolved_fault(agent, event)
            return
        if (
            event.cleared_fault not in navigation_faults
            or task.phase not in navigational_phases
            or agent.state.pose_estimate is None
            or agent.current_leg_goal is None
            or self.latest_snapshot is None
        ):
            self._restore_unresolved_fault(agent, event)
            return

        self._reset_recovery_robot_ids.add(event.robot_id)
        view = self.latest_snapshot.robot(event.robot_id)
        blocked = self._blocked_cells(event.robot_id, view.footprint_radius_m)
        if self._replan_agent(agent, agent.state.pose_estimate.pose, blocked):
            task.update_status(TaskStatus.IN_PROGRESS)
            agent.transition_to(NavigatingState(), event.sim_time)
            self._reset_recovery_robot_ids.discard(event.robot_id)
            return
        agent.route_replan_required = True
        self._route_blocked_robot_ids.add(event.robot_id)
        self._restore_unresolved_fault(agent, event)

    def _resume_reconciled_transfer(
        self, agent: ICoordinatedRobot, sim_time: float
    ) -> bool:
        """Resume only when observed payload state agrees with a safe phase."""
        task = agent.current_task
        estimate = agent.state.pose_estimate
        if task is None or estimate is None or self.latest_snapshot is None:
            return False

        payload_present = agent.state.payload_present
        if task.phase is TaskExecutionPhase.PICKING and payload_present is False:
            task.phase = TaskExecutionPhase.TO_SOURCE
        elif task.phase is TaskExecutionPhase.DROPPING and payload_present is True:
            task.phase = TaskExecutionPhase.TO_TARGET

        expected_presence_by_phase = {
            TaskExecutionPhase.TO_SOURCE: False,
            TaskExecutionPhase.TO_TARGET: True,
        }
        expected_presence = (
            None
            if task.phase is None
            else expected_presence_by_phase.get(task.phase)
        )
        if expected_presence is None or payload_present is not expected_presence:
            return False

        destination = (
            task.source_coords
            if task.phase is TaskExecutionPhase.TO_SOURCE
            else task.target_coords
        )
        if destination is None:
            return False
        view = self.latest_snapshot.robot(agent.state.robot_id)
        blocked = self._blocked_cells(
            agent.state.robot_id, view.footprint_radius_m
        )
        active_paths = tuple(
            robot.active_path
            for robot in self.latest_snapshot.robots
            if robot.robot_id != agent.state.robot_id
        )
        access, path = self.path_planner.find_path_to_access(
            estimate.pose,
            destination,
            self.topology,
            active_paths,
            blocked_cells=blocked,
            footprint_radius_m=agent.FOOTPRINT_RADIUS_M,
        )
        if access is None or not path:
            return False
        agent.current_leg_goal = access
        agent.set_path(path)
        agent.route_replan_required = False
        task.update_status(TaskStatus.IN_PROGRESS)
        agent.transition_to(NavigatingState(), sim_time)
        return True

    @staticmethod
    def _restore_unresolved_fault(
        agent: ICoordinatedRobot, event: FaultResetEvent
    ) -> None:
        """Keep an unresolved recovery retryable instead of faultless-faulted."""
        if agent.current_task is not None:
            agent.current_task.update_status(TaskStatus.RECOVERY_REQUIRED)
        agent.state.activate_faults(event.cleared_fault)
        agent.transition_to(FaultedState(), event.sim_time)

    def _dispatch_ready_tasks(self) -> None:
        ready_tasks = self.reservation_service.ready_queue
        idle_agents = [
            agent
            for agent in self.robots.values()
            if agent.fsm_status is FSMStatus.IDLE
            and agent.current_task is None
            and not agent.state.is_busy
            and agent.state.pose_estimate is not None
            and agent.state.localization_status
            in (LocalizationStatus.TRUSTED, LocalizationStatus.DEGRADED)
            and not agent.state.active_faults
        ]
        allocations = self.allocation_strategy.allocate_tasks(
            ready_tasks, [agent.state for agent in idle_agents]
        )
        assigned_tasks = []
        eligible_agent_ids = {
            agent.state.robot_id for agent in idle_agents
        }
        assigned_robot_ids = set()
        for task in tuple(ready_tasks):
            robot_id = allocations.get(task.task_id)
            if (
                robot_id in eligible_agent_ids
                and robot_id not in assigned_robot_ids
                and self._assign_and_route_task(
                    task, robot_id
                )
            ):
                assigned_tasks.append(task)
                assigned_robot_ids.add(robot_id)
        for task in assigned_tasks:
            ready_tasks.remove(task)

    def _assign_and_route_task(
        self,
        task: Task,
        agent_id: str,
    ) -> bool:
        agent = self.robots.get(agent_id)
        if agent is None or agent.state.pose_estimate is None:
            return False
        start_pose = agent.state.pose_estimate.pose
        if self.latest_snapshot is None:
            active_paths = tuple(
                tuple(other.path[other.path_index:])
                for other in self.robots.values()
                if other.path and other.state.robot_id != agent.state.robot_id
            )
            blocked_cells: FrozenSet[Tuple[int, int]] = frozenset()
        else:
            active_paths = tuple(
                view.active_path
                for view in self.latest_snapshot.robots
                if view.robot_id != agent.state.robot_id
            )
            blocked_cells = self._blocked_cells(
                agent.state.robot_id, agent.FOOTPRINT_RADIUS_M
            )

        destination = (
            task.source_coords
            if task.task_type
            in (TaskType.STORE, TaskType.RETRIEVE, TaskType.RELOCATE)
            else task.target_coords
        )
        if destination is None:
            return False
        target_access, path = self.path_planner.find_path_to_access(
            start_pose,
            destination,
            self.topology,
            active_paths,
            blocked_cells=blocked_cells,
            footprint_radius_m=agent.FOOTPRINT_RADIUS_M,
        )
        if target_access is None or not path:
            return False
        if (
            task.task_type is not TaskType.PARK
            and task.ideal_distance_m is None
        ):
            task.ideal_distance_m = self.baseline_estimator.estimate(
                task,
                start_pose,
                agent.FOOTPRINT_RADIUS_M,
                agent.WAYPOINT_TOLERANCE,
                agent.FINAL_WAYPOINT_TOLERANCE,
            )
        agent.assign_task(task)
        agent.current_leg_goal = target_access
        agent.set_path(path)
        return True

    def handle_robot_idle(self, event: RobotIdleEvent) -> None:
        """Dispatch queued work, otherwise return a free robot to parking."""
        if not tuple(self.robots.values()):
            return
        if self.reservation_service.ready_queue:
            self._dispatch_ready_tasks()
        agent = self.robots.get(event.robot_id)
        if agent is None or agent.current_task is not None:
            return
        current_grid = tuple(int(round(value)) for value in event.current_coords)
        if self.topology.get_zone_type(*current_grid) is ZoneType.PARKING:
            return
        park_task = Task.park(
            f"park_{uuid.uuid4().hex[:8]}", agent.state.home_base_coords
        )
        active_paths = (
            ()
            if self.latest_snapshot is None
            else tuple(
                view.active_path
                for view in self.latest_snapshot.robots
                if view.robot_id != event.robot_id
            )
        )
        estimate = agent.state.pose_estimate
        if estimate is None:
            return
        path = self.path_planner.find_path(
            estimate.pose,
            agent.state.home_base_coords,
            self.topology,
            active_paths,
            blocked_cells=self._blocked_cells(
                agent.state.robot_id, agent.FOOTPRINT_RADIUS_M
            ),
            footprint_radius_m=agent.FOOTPRINT_RADIUS_M,
        )
        if path:
            agent.assign_task(park_task)
            agent.set_path(path)

    def _blocked_cells(
        self, robot_id: str, footprint_radius_m: float
    ) -> FrozenSet[Tuple[int, int]]:
        if self.latest_snapshot is None:
            return frozenset()
        return blocked_cells_for_robot(
            self.latest_snapshot,
            self.topology,
            robot_id,
            footprint_radius_m,
        )

    def _replan_agent(
        self,
        agent: ICoordinatedRobot,
        start: Pose,
        blocked_cells: FrozenSet[Tuple[int, int]],
    ) -> bool:
        if self.latest_snapshot is None or agent.current_task is None:
            return False
        active_paths = tuple(
            robot.active_path
            for robot in self.latest_snapshot.robots
            if robot.robot_id != agent.state.robot_id
        )
        task = agent.current_task
        access: Optional[Tuple[int, int]]
        path: list[Tuple[int, int]]
        if task.task_type is TaskType.PARK:
            access = task.target_coords
            path = self.path_planner.find_path(
                start,
                access,
                self.topology,
                active_paths,
                blocked_cells=blocked_cells,
                footprint_radius_m=agent.FOOTPRINT_RADIUS_M,
            )
        else:
            destination = (
                task.source_coords
                if task.phase is TaskExecutionPhase.TO_SOURCE
                else task.target_coords
            )
            if destination is None:
                return False
            access, path = self.path_planner.find_path_to_access(
                start,
                destination,
                self.topology,
                active_paths,
                blocked_cells=blocked_cells,
                footprint_radius_m=agent.FOOTPRINT_RADIUS_M,
            )
        if access is None:
            return False
        if not path:
            return False
        agent.current_leg_goal = access
        agent.set_path(path)
        agent.route_replan_required = False
        if agent.fsm_status is FSMStatus.IDLE and agent.current_task is not None:
            agent.transition_to(NavigatingState(), self.latest_snapshot.sim_time)
        return True
