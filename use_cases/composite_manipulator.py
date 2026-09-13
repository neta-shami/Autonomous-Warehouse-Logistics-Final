from typing import Dict

from entities.enums import (
    CardinalDirection,
    ManipulatorAbortResult,
    ManipulatorResult,
)
from interfaces.i_extend_system import IExtendSystem
from interfaces.i_grip_system import IGripSystem
from interfaces.i_lift_system import ILiftSystem
from interfaces.i_manipulator_feedback import IManipulatorFeedback
from interfaces.i_manipulator_system import IManipulatorSystem
from interfaces.i_simulation_clock import ISimulationClock
from interfaces.i_turret_system import ITurretSystem
from use_cases.manipulator_operation import ManipulatorOperation


class CompositeManipulator(IManipulatorSystem):
    """Run feedback-gated pick and drop operations for independent robots."""

    TIER_HEIGHTS = {0: 0.38, 1: 0.78}
    CARRY_HEIGHT = 0.55
    FULL_EXTENSION = 0.5
    JOINT_TOLERANCE = 0.015
    TURRET_TOLERANCE = 0.03
    OPERATION_TIMEOUT_SECONDS = 3.0

    def __init__(
        self,
        turret_system: ITurretSystem,
        lift_system: ILiftSystem,
        extend_system: IExtendSystem,
        grip_system: IGripSystem,
        feedback: IManipulatorFeedback,
        clock: ISimulationClock,
    ):
        self.turret_system = turret_system
        self.lift_system = lift_system
        self.extend_system = extend_system
        self.grip_system = grip_system
        self.feedback = feedback
        self.clock = clock
        self.states: Dict[str, ManipulatorOperation] = {}

    def pick(
        self, robot_id: str, expected_payload_id: str, tier: int,
        approach: CardinalDirection
    ) -> ManipulatorResult:
        return self._run_sequence(
            robot_id, expected_payload_id, tier, approach, is_pick=True
        )

    def drop(
        self, robot_id: str, expected_payload_id: str, tier: int,
        approach: CardinalDirection
    ) -> ManipulatorResult:
        return self._run_sequence(
            robot_id, expected_payload_id, tier, approach, is_pick=False
        )

    def abort(self, robot_id: str) -> ManipulatorAbortResult:
        """Retract an active operation without publishing inventory events."""
        operation = self.states.pop(robot_id, None)
        if operation is None:
            return ManipulatorAbortResult.CLEAN

        self.extend_system.extend_arm(robot_id, 0.0)
        safe_height = (
            self.CARRY_HEIGHT
            if operation.is_pick and operation.committed
            else 0.0
        )
        self.lift_system.set_lift_height(robot_id, safe_height)
        if operation.committed:
            return ManipulatorAbortResult.RECONCILIATION_REQUIRED
        return ManipulatorAbortResult.CLEAN

    def _run_sequence(
        self,
        robot_id: str,
        expected_payload_id: str,
        tier: int,
        approach: CardinalDirection,
        is_pick: bool,
    ) -> ManipulatorResult:
        if not isinstance(expected_payload_id, str) or not expected_payload_id:
            raise ValueError("expected_payload_id must be a non-empty string")
        tier_height = self._tier_height(tier)
        now = self.clock.get_time()
        operation = self.states.get(robot_id)
        if operation is None:
            operation = ManipulatorOperation(
                is_pick=is_pick,
                expected_payload_id=expected_payload_id,
                tier=tier,
                approach=approach,
                started_at=now,
            )
            self.states[robot_id] = operation
        elif not operation.matches(
            is_pick, expected_payload_id, tier, approach
        ):
            raise ValueError(
                f"Robot '{robot_id}' already has an active manipulation "
                "with different parameters"
            )

        elapsed = now - operation.started_at
        if elapsed < 0.0 or elapsed >= self.OPERATION_TIMEOUT_SECONDS:
            return ManipulatorResult.FAILED

        self.turret_system.aim_turret(robot_id, approach)
        if operation.stage == "position":
            self.lift_system.set_lift_height(robot_id, tier_height)
            if self._joint_ready(
                robot_id,
                "turret",
                approach.angle_radians,
                self.TURRET_TOLERANCE,
            ) and self._joint_ready(
                robot_id, "lift", tier_height, self.JOINT_TOLERANCE
            ):
                operation.stage = "extend"

        elif operation.stage == "extend":
            self.lift_system.set_lift_height(robot_id, tier_height)
            self.extend_system.extend_arm(robot_id, self.FULL_EXTENSION)
            if (
                self._joint_ready(
                    robot_id,
                    "turret",
                    approach.angle_radians,
                    self.TURRET_TOLERANCE,
                )
                and self._joint_ready(
                    robot_id, "lift", tier_height, self.JOINT_TOLERANCE
                )
                and self._joint_ready(
                    robot_id,
                    "extend",
                    self.FULL_EXTENSION,
                    self.JOINT_TOLERANCE,
                )
            ):
                operation.stage = "commit"

        elif operation.stage == "commit":
            self.grip_system.grip(
                robot_id, expected_payload_id, engage=is_pick
            )
            operation.committed = True
            operation.stage = "retract"

        elif operation.stage == "retract":
            self.lift_system.set_lift_height(robot_id, tier_height)
            self.extend_system.extend_arm(robot_id, 0.0)
            if self._joint_ready(
                robot_id, "extend", 0.0, self.JOINT_TOLERANCE
            ):
                operation.stage = "finish"

        elif operation.stage == "finish":
            final_height = self.CARRY_HEIGHT if is_pick else 0.0
            self.lift_system.set_lift_height(robot_id, final_height)
            if self._joint_ready(
                robot_id, "lift", final_height, self.JOINT_TOLERANCE
            ):
                del self.states[robot_id]
                return ManipulatorResult.SUCCEEDED

        return ManipulatorResult.IN_PROGRESS

    def _joint_ready(
        self,
        robot_id: str,
        joint_name: str,
        target: float,
        tolerance: float,
    ) -> bool:
        return self.feedback.joint_at_target(
            robot_id, joint_name, target, tolerance
        )

    def _tier_height(self, tier: int) -> float:
        try:
            return self.TIER_HEIGHTS[tier]
        except KeyError as exc:
            raise ValueError(f"Unsupported shelf tier: {tier}") from exc
