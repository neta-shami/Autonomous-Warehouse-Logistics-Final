from typing import Dict, Tuple, Union


class GridStateSnapshot:
    def __init__(self):
        # Maps (x, y, tier) to either a boolean (occupied) or string (item_id)
        self.occupancy_map: Dict[Tuple[int, int, int], Union[bool, str]] = {}

    def is_cell_empty(self, x: int, y: int, tier: int) -> bool:
        return (x, y, tier) not in self.occupancy_map or not self.occupancy_map[(x, y, tier)]

    def get_cell_occupancy(self, x: int, y: int, tier: int):
        """Return the confirmed occupant id/value, or None when empty."""
        return self.occupancy_map.get((x, y, tier))

    def set_cell_occupancy(self, x: int, y: int, tier: int, is_full: Union[bool, str]) -> None:
        if is_full:
            self.occupancy_map[(x, y, tier)] = is_full
        else:
            self.occupancy_map.pop((x, y, tier), None)
