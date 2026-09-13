"""Benchmark per-tick wall-localization and fusion scaling."""

from statistics import median
from time import perf_counter

from entities.drive_command import DriveCommand
from entities.enums import CardinalDirection, LocalizationStatus, PoseSource
from entities.pose import Pose
from entities.pose_estimate import PoseEstimate
from entities.sensor_frame import SensorFrame
from entities.warehouse_topology import WarehouseTopology
from use_cases.continuous_wall_range_localizer import (
    ContinuousWallRangeLocalizer,
)
from use_cases.holonomic_command_motion_model import (
    HolonomicCommandMotionModel,
)
from use_cases.pose_fusion_service import PoseFusionService

FLEET_SIZES = (1, 10, 100, 500)
ROUNDS = 200


def _frame(robot_id: str) -> SensorFrame:
    return SensorFrame(
        robot_id,
        (0.0, 0.0),
        (
            (CardinalDirection.POS_X, 8.0),
            (CardinalDirection.NEG_X, 2.0),
            (CardinalDirection.POS_Y, 7.0),
            (CardinalDirection.NEG_Y, 3.0),
        ),
        -1.0,
        1.0,
        1,
    )


def benchmark(fleet_size: int) -> tuple[float, float]:
    topology = WarehouseTopology(10, 10, 0.0, 10.0, 0.0, 10.0, {})
    localizer = ContinuousWallRangeLocalizer(0.05, 0.01)
    services = [
        PoseFusionService(
            HolonomicCommandMotionModel(), localizer, topology
        )
        for _ in range(fleet_size)
    ]
    frames = [_frame(f"r{index}") for index in range(fleet_size)]
    estimates = [
        PoseEstimate(Pose(2.0, 3.0), 0.02, PoseSource.FUSED, 0.98)
        for _ in range(fleet_size)
    ]
    samples = []
    for _ in range(ROUNDS):
        started = perf_counter()
        for service, frame, estimate in zip(services, frames, estimates):
            service.assess(
                frame,
                estimate,
                LocalizationStatus.TRUSTED,
                DriveCommand(0.0, 0.0),
                0.02,
            )
        samples.append(perf_counter() - started)
    tick_seconds = median(samples)
    return tick_seconds * 1000.0, tick_seconds * 1_000_000.0 / fleet_size


def main() -> None:
    print("fleet_size,median_tick_ms,median_per_robot_us")
    for fleet_size in FLEET_SIZES:
        tick_ms, robot_us = benchmark(fleet_size)
        print(f"{fleet_size},{tick_ms:.4f},{robot_us:.4f}")
    print(
        "Note: this measures Python frame interpretation and fusion only. "
        "MuJoCo sensor evaluation occurs inside every physics step and is not "
        "divided by the 50 Hz Python logic rate."
    )


if __name__ == "__main__":
    main()
