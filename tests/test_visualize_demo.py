"""Interactive demo configuration tests that do not launch a GUI."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest

from adapters.application_factory import build_application
from adapters.simulation_loader import SimulationLoader
from entities.enums import TaskType
from entities.events import TaskCompletedEvent
from visualize import (
    VisualizationDemoWorkload,
    build_contention_demo_tasks,
    configure_initial_view,
    hide_rangefinder_rays,
)


def _topology():
    xml_path = Path(__file__).parents[1] / "warehouse.xml"
    _, _, topology = SimulationLoader(str(xml_path)).load()
    return topology


def test_contention_tasks_include_one_explicit_duplicate_request():
    primary, duplicate, independent = build_contention_demo_tasks(_topology())

    assert primary.task_type is TaskType.RELOCATE
    assert duplicate.payload_id == primary.payload_id == "p4"
    assert duplicate.source_coords == primary.source_coords
    assert duplicate.target_coords != primary.target_coords
    assert independent.payload_id == "p3"


def test_demo_starts_every_operation_and_blocks_duplicate_package():
    runtime = build_application(
        str(Path(__file__).resolve().parents[1] / "warehouse.xml")
    )
    workload = VisualizationDemoWorkload(runtime)

    workload.start()

    assert Counter(task.task_type for task in runtime.reservations.ready_queue) == {
        TaskType.RELOCATE: 2,
        TaskType.RETRIEVE: 1,
        TaskType.STORE: 1,
    }
    assert [task.task_id for task in runtime.reservations.blocked_queue] == [
        "demo_p4_conflict"
    ]
    assert runtime.inventory.task_leases["demo_p4_primary"].payload_id == "p4"
    assert "demo_p4_conflict" not in runtime.inventory.task_leases


def test_first_mixed_stage_uses_all_four_physical_robots_safely():
    runtime = build_application(
        str(Path(__file__).resolve().parents[1] / "warehouse.xml")
    )
    completed_types = []

    def record_completion(event):
        if event.success and event.task and event.task.task_type is not TaskType.PARK:
            completed_types.append(event.task.task_type)

    runtime.orchestrator.event_dispatcher.subscribe(
        TaskCompletedEvent, record_completion
    )
    workload = VisualizationDemoWorkload(runtime)
    workload.start()

    for _ in range(5_000):
        runtime.orchestrator.step()
        workload.advance_if_settled()
        if workload.stage == 1:
            break

    assert runtime.metrics.total_throughput == 4
    assert Counter(completed_types) == {
        TaskType.RELOCATE: 2,
        TaskType.RETRIEVE: 1,
        TaskType.STORE: 1,
    }
    assert runtime.metrics.successful_robot_ids == {"r1", "r2", "r3", "r4"}
    assert runtime.metrics.total_failed_tasks == 0
    assert runtime.metrics.total_actual_collisions == 0
    assert runtime.metrics.unsafe_command_batches == 0
    assert runtime.metrics.peak_orphan_lease_count == 0
    assert [task.task_id for task in runtime.reservations.blocked_queue] == [
        "demo_p4_conflict"
    ]


def test_staged_demo_repeats_move_retrieve_and_store_with_all_robots():
    runtime = build_application(
        str(Path(__file__).resolve().parents[1] / "warehouse.xml")
    )
    completed_types = []

    def record_completion(event):
        if event.success and event.task and event.task.task_type is not TaskType.PARK:
            completed_types.append(event.task.task_type)

    runtime.orchestrator.event_dispatcher.subscribe(
        TaskCompletedEvent, record_completion
    )
    workload = VisualizationDemoWorkload(runtime)
    workload.start()

    for _ in range(10_000):
        runtime.orchestrator.step()
        workload.advance_if_settled()
        if workload.complete:
            break

    assert workload.complete
    assert Counter(completed_types) == {
        TaskType.RELOCATE: 4,
        TaskType.RETRIEVE: 2,
        TaskType.STORE: 2,
    }
    assert runtime.metrics.successful_robot_ids == {"r1", "r2", "r3", "r4"}
    assert runtime.metrics.total_actual_collisions == 0
    assert runtime.metrics.unsafe_command_batches == 0
    assert runtime.metrics.peak_orphan_lease_count == 0


def test_visualizer_hides_rangefinder_overlay_without_changing_model():
    viewer = SimpleNamespace(opt=mujoco.MjvOption())
    flag = int(mujoco.mjtVisFlag.mjVIS_RANGEFINDER)
    assert viewer.opt.flags[flag]

    hide_rangefinder_rays(viewer)

    assert not viewer.opt.flags[flag]


def test_visualizer_starts_with_complete_warehouse_in_diagonal_view():
    topology = _topology()
    viewer = SimpleNamespace(cam=mujoco.MjvCamera())

    configure_initial_view(viewer, topology)

    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert viewer.cam.fixedcamid == -1
    assert viewer.cam.trackbodyid == -1
    assert tuple(viewer.cam.lookat) == (5.0, 5.0, 0.5)
    assert viewer.cam.distance == pytest.approx(16.0)
    assert viewer.cam.azimuth == 90.0
    assert viewer.cam.elevation == -45.0
