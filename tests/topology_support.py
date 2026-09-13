"""Small logical warehouse used by inventory-focused tests."""

from entities.enums import ZoneType
from entities.warehouse_topology import WarehouseTopology


def sample_topology(*extra_shelves):
    zone_map = {
        (2, 2): ZoneType.SHELF,
        (2, 5): ZoneType.SHELF,
        (4, 4): ZoneType.SHELF,
        (5, 5): ZoneType.SHELF,
        (6, 6): ZoneType.SHELF,
        (9, 1): ZoneType.INBOUND_DOCK,
        (9, 8): ZoneType.OUTBOUND_DOCK,
        (3, 9): ZoneType.PARKING,
    }
    zone_map.update(dict.fromkeys(extra_shelves, ZoneType.SHELF))
    return WarehouseTopology(10, 10, 0.0, 10.0, 0.0, 10.0, zone_map)
