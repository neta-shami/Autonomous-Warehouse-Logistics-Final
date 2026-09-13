import pytest

from entities.enums import CardinalDirection, ZoneType
from entities.warehouse_topology import WarehouseTopology
from use_cases.astar_pathfinding_strategy import AStarPathfindingStrategy
from use_cases.grid_path_planner import GridPathPlanner


@pytest.fixture
def shelf_topology():
    return WarehouseTopology(
        5, 5, 0.0, 5.0, 0.0, 5.0, {(2, 2): ZoneType.SHELF}
    )


@pytest.mark.parametrize(
    ("robot", "access", "direction"),
    [
        ((0, 2), (1, 2), CardinalDirection.POS_X),
        ((4, 2), (3, 2), CardinalDirection.NEG_X),
        ((2, 0), (2, 1), CardinalDirection.POS_Y),
        ((2, 4), (2, 3), CardinalDirection.NEG_Y),
    ],
)
def test_selects_nearest_shelf_access_and_direction(
    shelf_topology, robot, access, direction
):
    assert shelf_topology.get_access_point((2, 2), robot) == access
    assert shelf_topology.get_approach_direction((2, 2), access) is direction


def test_dock_is_accessed_from_an_adjacent_cell():
    topology = WarehouseTopology(
        10, 10, 0.0, 10.0, 0.0, 10.0, {(9, 1): ZoneType.INBOUND_DOCK}
    )

    access = topology.get_access_point((9, 1), (9, 9))

    assert access == (9, 2)
    assert not topology.is_navigable(9, 1)
    assert topology.get_approach_direction((9, 1), access) is CardinalDirection.NEG_Y


def test_planner_uses_an_alternate_access_when_nearest_side_is_blocked(
    shelf_topology,
):
    planner = GridPathPlanner(AStarPathfindingStrategy())

    access, path = planner.find_path_to_access(
        (0, 2),
        (2, 2),
        shelf_topology,
        (),
        blocked_cells=frozenset({(1, 2)}),
    )

    assert access in {(2, 1), (2, 3), (3, 2)}
    assert path
    assert path[-1] == access
    assert (1, 2) not in path


def test_parking_cell_remains_the_route_destination():
    topology = WarehouseTopology(
        5, 5, 0.0, 5.0, 0.0, 5.0, {(2, 4): ZoneType.PARKING}
    )
    assert topology.get_access_point((2, 4), (0, 0)) == (2, 4)


def test_fails_when_station_has_no_navigable_access_cell():
    blocked = {
        (2, 2): ZoneType.SHELF,
        (1, 2): ZoneType.SHELF,
        (3, 2): ZoneType.SHELF,
        (2, 1): ZoneType.SHELF,
        (2, 3): ZoneType.SHELF,
    }
    topology = WarehouseTopology(5, 5, 0.0, 5.0, 0.0, 5.0, blocked)

    with pytest.raises(ValueError, match="no navigable access point"):
        topology.get_access_point((2, 2), (0, 0))


@pytest.mark.parametrize("access", [(2, 2), (0, 0), (3, 3)])
def test_rejects_non_adjacent_approach(access):
    topology = WarehouseTopology(5, 5, 0.0, 5.0, 0.0, 5.0, {})

    with pytest.raises(ValueError, match="cardinally adjacent"):
        topology.get_approach_direction((2, 2), access)
