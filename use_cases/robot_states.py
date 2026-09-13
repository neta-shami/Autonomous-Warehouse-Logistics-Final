"""Robot State-pattern behaviors operating only on committed beliefs."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from entities.drive_command import DriveCommand
from entities.enums import (
    FaultCode,
    FSMStatus,
    LocalizationStatus,
    TaskExecutionPhase,
    TaskType,
)
from entities.events import (
    CollisionStallEvent,
    RelocalizationFailedEvent,
    RelocalizedEvent,
    RobotFaultedEvent,
    RobotIdleEvent,
)
from entities.runtime_control import TickContext

if TYPE_CHECKING:
    from use_cases.robot_agent import RobotAgent


class IRobotState(ABC):
    """Behavior contract for one robot lifecycle state."""

    @property
    @abstractmethod
    def fsm_status(self) -> FSMStatus:
        raise NotImplementedError

    @abstractmethod
    def update(self, agent: "RobotAgent", context: TickContext) -> None:
        raise NotImplementedError


class IdleState(IRobotState):
    def __init__(self):
        self._idle_event_fired = False

    @property
    def fsm_status(self) -> FSMStatus:
        return FSMStatus.IDLE

    def update(self, agent: "RobotAgent", context: TickContext) -> None:
        if not self._idle_event_fired:
            current_coords = (
                int(round(agent.state.current_coords[0])),
                int(round(agent.state.current_coords[1])),
            )
            agent.event_bus.append(
                RobotIdleEvent(
                    sim_time=context.sim_time,
                    robot_id=agent.state.robot_id,
                    current_coords=current_coords,
                )
            )
            self._idle_event_fired = True


class NavigatingState(IRobotState):
    @property
    def fsm_status(self) -> FSMStatus:
        return FSMStatus.NAVIGATING

    def update(self, agent: "RobotAgent", context: TickContext) -> None:
        if (
            agent.state.pose_estimate is None
            or agent.state.localization_status
            not in (LocalizationStatus.TRUSTED, LocalizationStatus.DEGRADED)
        ):
            agent.stop_drive()
            return
        if (
            agent.path
            and isinstance(agent.current_leg_goal, tuple)
            and agent.path[-1] != agent.current_leg_goal
        ):
            agent.stop_drive()
            return
        new_status, blocking_ids = agent.local_avoidance.check_collisions(
            agent.state.robot_id, context.fleet_snapshot
        )
        if new_status is FSMStatus.AVOIDING:
            agent.event_bus.append(
                CollisionStallEvent(
                    sim_time=context.sim_time,
                    robot_id=agent.state.robot_id,
                    coords=agent.state.current_coords,
                    blocking_robot_ids=list(blocking_ids),
                )
            )
            agent.stop_drive()
            agent.transition_to(AvoidingState())
            return

        if not agent.path or agent.path_index >= len(agent.path):
            agent.stop_drive()
            self._enter_manipulation(agent)
            return

        while agent.path_index < len(agent.path):
            target = agent.path[agent.path_index]
            is_final = agent.path_index == len(agent.path) - 1
            tolerance = (
                agent.FINAL_WAYPOINT_TOLERANCE
                if is_final
                else agent.WAYPOINT_TOLERANCE
            )
            command = agent.waypoint_controller.command_for(
                agent.state.pose_estimate.pose, target, tolerance
            )
            if command != DriveCommand(0.0, 0.0):
                agent.apply_drive(command)
                return
            agent.path_index += 1

        agent.stop_drive()
        self._enter_manipulation(agent)

    @staticmethod
    def _enter_manipulation(agent: "RobotAgent") -> None:
        task = agent.current_task
        if task is not None:
            if task.task_type is TaskType.PARK:
                task.phase = TaskExecutionPhase.COMPLETE
            elif task.phase is TaskExecutionPhase.TO_SOURCE:
                task.phase = TaskExecutionPhase.PICKING
            elif task.phase is TaskExecutionPhase.TO_TARGET:
                task.phase = TaskExecutionPhase.DROPPING
        agent.transition_to(ManipulatingState())


class AvoidingState(IRobotState):
    @property
    def fsm_status(self) -> FSMStatus:
        return FSMStatus.AVOIDING

    def update(self, agent: "RobotAgent", context: TickContext) -> None:
        agent.stop_drive()
        new_status, _ = agent.local_avoidance.check_collisions(
            agent.state.robot_id, context.fleet_snapshot
        )
        if new_status is None:
            agent.transition_to(NavigatingState())


class ManipulatingState(IRobotState):
    @property
    def fsm_status(self) -> FSMStatus:
        return FSMStatus.MANIPULATING

    def update(self, agent: "RobotAgent", context: TickContext) -> None:
        agent.stop_drive()
        if not agent.current_task or not agent.task_strategy:
            agent.state.is_busy = False
            agent.transition_to(IdleState())
            return

        task_completed = agent.task_strategy.execute(agent, context.sim_time)
        if task_completed:
            agent.current_task = None
            agent.task_strategy = None
            agent.path = []
            agent.path_index = 0
            agent.route_replan_required = False
            agent.state.is_busy = False
            agent.state.expected_payload_id = None
            agent.current_leg_goal = None
            agent.transition_to(IdleState())


class RelocalizingState(IRobotState):
    """Command-free state used until a trusted sensor fix is committed."""

    TIMEOUT_SECONDS = 2.0

    @property
    def fsm_status(self) -> FSMStatus:
        return FSMStatus.RELOCALIZING

    def update(self, agent: "RobotAgent", context: TickContext) -> None:
        if agent.state_start_sim_time is None:
            agent.state_start_sim_time = context.sim_time
        if (
            agent.state.localization_status is LocalizationStatus.TRUSTED
            and agent.state.pose_estimate is not None
            and not agent.state.active_faults
        ):
            if agent._relocalizing_due_to_loss:
                agent.path = []
                agent.path_index = 0
                agent.event_bus.append(
                    RelocalizedEvent(
                        context.sim_time,
                        agent.state.robot_id,
                        agent.state.pose_estimate.pose,
                        context.sim_time - agent.state_start_sim_time,
                    )
                )
                agent._relocalizing_due_to_loss = False
            agent.transition_to(IdleState(), context.sim_time)
            return
        if context.sim_time - agent.state_start_sim_time >= self.TIMEOUT_SECONDS:
            agent.state.activate_faults(FaultCode.STATE_TIMEOUT)
            agent.event_bus.append(
                RelocalizationFailedEvent(context.sim_time, agent.state.robot_id)
            )
            agent.transition_to(FaultedState(), context.sim_time)


class FaultedState(IRobotState):
    """Command-free placeholder; Task 1.10 adds full recovery policy."""

    def __init__(self):
        self._event_fired = False

    @property
    def fsm_status(self) -> FSMStatus:
        return FSMStatus.FAULTED

    def update(self, agent: "RobotAgent", context: TickContext) -> None:
        if self._event_fired:
            return
        agent.event_bus.append(
            RobotFaultedEvent(
                context.sim_time,
                agent.state.robot_id,
            )
        )
        self._event_fired = True
