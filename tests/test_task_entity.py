import pytest

from entities.enums import TaskExecutionPhase, TaskType
from entities.task import Task


def test_transfer_factory_exposes_recoverable_initial_phase_and_payload():
    task = Task.transfer(
        "retrieve", TaskType.RETRIEVE, (2, 5), 0, (9, 8), 0, "p2"
    )

    assert task.payload_id == "p2"
    assert task.phase is TaskExecutionPhase.TO_SOURCE
    assert task.lease_generation == 0


def test_park_factory_has_only_a_target_leg():
    task = Task.park("park", (1, 9))

    assert task.source_coords is None
    assert task.source_tier is None
    assert task.payload_id is None
    assert task.phase is TaskExecutionPhase.TO_TARGET


def test_transfer_factory_rejects_park_type():
    with pytest.raises(ValueError, match="transfer task type"):
        Task.transfer("bad", TaskType.PARK, (1, 1), 0, (2, 2), 0)


def test_task_coordinates_must_be_integer_pairs():
    with pytest.raises(TypeError, match="target_coords"):
        Task.park("bad", (1.5, 2))
