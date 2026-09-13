"""Contract tests for the single production composition root."""

from pathlib import Path

from adapters.application_factory import DEFAULT_ROBOTS, build_application
from entities.enums import SnapshotType, TrafficPattern


def test_application_factory_wires_one_registry_and_four_physical_robots():
    runtime = build_application(
        str(Path(__file__).resolve().parents[1] / "warehouse.xml")
    )

    expected_ids = tuple(sorted(spec.robot_id for spec in DEFAULT_ROBOTS))
    assert runtime.robots.robot_ids() == expected_ids
    assert runtime.fleet.robots is runtime.robots
    assert runtime.traffic.robots is runtime.robots
    assert runtime.orchestrator.robot_registry is runtime.robots
    assert set(runtime.traffic.strategies) == {
        TrafficPattern.REAR_END,
        TrafficPattern.HEAD_ON,
        TrafficPattern.CROSSING,
        TrafficPattern.STUCK_BETWEEN,
    }


def test_application_factory_initial_inventory_matches_physical_model():
    runtime = build_application(
        str(Path(__file__).resolve().parents[1] / "warehouse.xml")
    )
    physical = runtime.inventory.get_snapshot(SnapshotType.PHYSICAL)
    reserved = runtime.inventory.get_snapshot(SnapshotType.RESERVED)

    for slot, package_id in {
        (2, 5, 0): "p2",
        (4, 4, 1): "p3",
        (6, 6, 0): "p4",
        (8, 3, 0): "p6",
    }.items():
        assert physical.get_cell_occupancy(*slot) == package_id
        assert reserved.get_cell_occupancy(*slot) == package_id
