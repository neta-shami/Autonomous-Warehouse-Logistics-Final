import pytest

from adapters.cli_parser import CLIParser
from entities.enums import TaskExecutionPhase, TaskType, ZoneType
from entities.warehouse_topology import WarehouseTopology

TOPOLOGY = WarehouseTopology(
    10,
    10,
    0.0,
    10.0,
    0.0,
    10.0,
    {
        (2, 2): ZoneType.SHELF,
        (2, 5): ZoneType.SHELF,
        (4, 4): ZoneType.SHELF,
        (6, 6): ZoneType.SHELF,
        (9, 1): ZoneType.INBOUND_DOCK,
        (9, 8): ZoneType.OUTBOUND_DOCK,
    },
)


def test_parses_retrieve():
    tasks = CLIParser.parse_commands(["RETRIEVE p2 2,5 0"], TOPOLOGY)
    assert len(tasks) == 1
    assert tasks[0].task_type is TaskType.RETRIEVE
    assert tasks[0].source_coords == (2, 5)
    assert tasks[0].source_tier == 0
    assert tasks[0].payload_id == "p2"
    assert tasks[0].phase is TaskExecutionPhase.TO_SOURCE


def test_parses_store():
    tasks = CLIParser.parse_commands(["STORE p1 2,2 1"], TOPOLOGY)
    assert len(tasks) == 1
    assert tasks[0].task_type is TaskType.STORE
    assert tasks[0].target_coords == (2, 2)
    assert tasks[0].target_tier == 1
    assert tasks[0].payload_id == "p1"


def test_parses_move_with_both_tiers():
    tasks = CLIParser.parse_commands(["MOVE 4,4 1 6,6 0"], TOPOLOGY)
    assert len(tasks) == 1
    assert tasks[0].task_type is TaskType.RELOCATE
    assert tasks[0].source_coords == (4, 4)
    assert tasks[0].source_tier == 1
    assert tasks[0].target_coords == (6, 6)
    assert tasks[0].target_tier == 0
    assert tasks[0].payload_id is None


def test_unknown_command_is_reported_not_swallowed(capsys):
    """Regression test: unrecognised verbs fell through silently, so a typo in
    a scripted command list produced a run that did nothing and said nothing.
    """
    tasks = CLIParser.parse_commands(["PICK 5,5 1 2,2 1"], TOPOLOGY)
    assert tasks == []
    assert "Unknown command 'PICK'" in capsys.readouterr().out


def test_malformed_command_is_reported(capsys):
    tasks = CLIParser.parse_commands(["MOVE 1,1 5,5"], TOPOLOGY)
    assert tasks == []
    assert "Failed to parse" in capsys.readouterr().out


def test_blank_lines_are_skipped():
    assert CLIParser.parse_commands(["", "   "], TOPOLOGY) == []


def test_one_bad_command_does_not_lose_the_good_ones():
    tasks = CLIParser.parse_commands(
        ["RETRIEVE p2 2,5 0", "NONSENSE", "STORE p1 2,2 1"],
        TOPOLOGY,
    )
    assert [t.task_type for t in tasks] == [TaskType.RETRIEVE, TaskType.STORE]


@pytest.mark.parametrize(
    "command",
    [
        "STORE p1 3,3 0",
        "STORE p1 20,20 0",
        "STORE p1 2,2 9",
        "RETRIEVE p1 3,3 0",
        "MOVE 2,2 0 3,3 0",
    ],
)
def test_semantically_invalid_warehouse_locations_are_rejected(command, capsys):
    assert CLIParser.parse_commands([command], TOPOLOGY) == []
    assert "Failed to parse" in capsys.readouterr().out
