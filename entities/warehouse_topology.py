from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Dict, FrozenSet, Tuple

from .enums import CardinalDirection, TaskType, ZoneType
from .value_validation import require_finite

if TYPE_CHECKING:
    from .pose import Pose

@dataclass
class WarehouseTopology:
    SUPPORTED_SHELF_TIERS: ClassVar[FrozenSet[int]] = frozenset({0, 1})
    MANIPULATION_ZONES: ClassVar[FrozenSet[ZoneType]] = frozenset(
        {
            ZoneType.SHELF,
            ZoneType.INBOUND_DOCK,
            ZoneType.OUTBOUND_DOCK,
        }
    )
    _ACCESS_OFFSETS: ClassVar[Tuple[Tuple[int, int], ...]] = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )
    _ZONE_HALF_EXTENTS: ClassVar[Dict[ZoneType, Tuple[float, float]]] = {
        ZoneType.SHELF: (0.3, 0.4),
        ZoneType.INBOUND_DOCK: (0.3, 0.3),
        ZoneType.OUTBOUND_DOCK: (0.3, 0.3),
    }

    grid_width: int
    grid_height: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    zone_map: Dict[Tuple[int, int], ZoneType]

    def __post_init__(self) -> None:
        if (
            isinstance(self.grid_width, bool)
            or not isinstance(self.grid_width, int)
            or self.grid_width <= 0
            or isinstance(self.grid_height, bool)
            or not isinstance(self.grid_height, int)
            or self.grid_height <= 0
        ):
            raise ValueError("grid dimensions must be positive integers")
        for name, value in (
            ("min_x", self.min_x),
            ("max_x", self.max_x),
            ("min_y", self.min_y),
            ("max_y", self.max_y),
        ):
            require_finite(name, value)
        if self.max_x <= self.min_x or self.max_y <= self.min_y:
            raise ValueError("metric maximum bounds must exceed minimum bounds")

    def is_within_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

    def is_navigable(self, x: int, y: int) -> bool:
        """Return whether the chassis can occupy a cell.

        Shelves and package docks are physical stations. Robots stop beside
        them and let the arm enter their cell.
        """
        if not self.is_within_bounds(x, y):
            return False
        return self.get_zone_type(x, y) not in self.MANIPULATION_ZONES

    def get_zone_type(self, x: int, y: int) -> ZoneType:
        return self.zone_map.get((x, y), ZoneType.NORMAL_FLOOR)

    def validate_shelf_slot(
        self,
        coords: Tuple[int, int],
        tier: int,
        *,
        field_name: str = "shelf slot",
    ) -> None:
        """Reject task endpoints that cannot represent a physical shelf slot."""
        if (
            not isinstance(coords, tuple)
            or len(coords) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in coords
            )
        ):
            raise TypeError(
                f"{field_name} coordinates must be an (x, y) integer tuple"
            )
        x, y = coords
        if not self.is_within_bounds(x, y):
            raise ValueError(f"{field_name} {coords} is outside the warehouse grid")
        if self.get_zone_type(x, y) is not ZoneType.SHELF:
            raise ValueError(f"{field_name} {coords} is not a shelf")
        if isinstance(tier, bool) or tier not in self.SUPPORTED_SHELF_TIERS:
            supported = ", ".join(
                str(value) for value in sorted(self.SUPPORTED_SHELF_TIERS)
            )
            raise ValueError(
                f"{field_name} tier {tier!r} is unsupported; expected one of: {supported}"
            )

    def validate_task_locations(self, task) -> None:
        """Validate task endpoints against the warehouse's physical zones."""
        from entities.task import Task

        if not isinstance(task, Task):
            raise TypeError("task must be a Task")
        if task.task_type is TaskType.PARK:
            self._validate_endpoint(
                task.target_coords,
                task.target_tier,
                ZoneType.PARKING,
                "parking target",
            )
            return

        source_zone, target_zone = {
            TaskType.STORE: (ZoneType.INBOUND_DOCK, ZoneType.SHELF),
            TaskType.RETRIEVE: (ZoneType.SHELF, ZoneType.OUTBOUND_DOCK),
            TaskType.RELOCATE: (ZoneType.SHELF, ZoneType.SHELF),
        }[task.task_type]
        if task.source_coords is None or task.source_tier is None:
            raise ValueError("transfer task requires a source endpoint")
        self._validate_endpoint(
            task.source_coords, task.source_tier, source_zone, "task source"
        )
        self._validate_endpoint(
            task.target_coords, task.target_tier, target_zone, "task target"
        )

    def _validate_endpoint(
        self,
        coords: Tuple[int, int],
        tier: int,
        expected_zone: ZoneType,
        field_name: str,
    ) -> None:
        x, y = coords
        if not self.is_within_bounds(x, y):
            raise ValueError(f"{field_name} {coords} is outside the warehouse grid")
        actual_zone = self.get_zone_type(x, y)
        if actual_zone is not expected_zone:
            raise ValueError(
                f"{field_name} {coords} must be {expected_zone.name}, "
                f"not {actual_zone.name}"
            )
        supported_tiers = (
            self.SUPPORTED_SHELF_TIERS
            if expected_zone is ZoneType.SHELF
            else frozenset({0})
        )
        if tier not in supported_tiers:
            supported = ", ".join(str(value) for value in sorted(supported_tiers))
            raise ValueError(
                f"{field_name} tier {tier!r} is unsupported; expected one of: "
                f"{supported}"
            )

    def is_pose_feasible(
        self, pose: "Pose", footprint_radius_m: float
    ) -> bool:
        """Check a continuous circular chassis footprint against static space."""
        from entities.pose import Pose

        if not isinstance(pose, Pose):
            raise TypeError("pose must be a Pose")
        require_finite(
            "footprint_radius_m", footprint_radius_m, minimum=0.0
        )
        if (
            pose.x - footprint_radius_m < self.min_x
            or pose.x + footprint_radius_m > self.max_x
            or pose.y - footprint_radius_m < self.min_y
            or pose.y + footprint_radius_m > self.max_y
        ):
            return False

        for (x, y), zone in self.zone_map.items():
            if zone not in self.MANIPULATION_ZONES:
                continue
            half_x, half_y = self._ZONE_HALF_EXTENTS[zone]
            closest_x = min(max(pose.x, x - half_x), x + half_x)
            closest_y = min(max(pose.y, y - half_y), y + half_y)
            dx = pose.x - closest_x
            dy = pose.y - closest_y
            if dx * dx + dy * dy <= footprint_radius_m * footprint_radius_m:
                return False
        return True

    def is_segment_feasible(
        self,
        start: "Pose",
        end: "Pose",
        footprint_radius_m: float,
    ) -> bool:
        """Conservatively validate a straight continuous-footprint connector."""
        from entities.pose import Pose

        if not isinstance(start, Pose) or not isinstance(end, Pose):
            raise TypeError("connector endpoints must be Pose values")
        require_finite(
            "footprint_radius_m", footprint_radius_m, minimum=0.0
        )
        if not self.is_pose_feasible(
            start, footprint_radius_m
        ) or not self.is_pose_feasible(end, footprint_radius_m):
            return False

        for (x, y), zone in self.zone_map.items():
            if zone not in self.MANIPULATION_ZONES:
                continue
            half_x, half_y = self._ZONE_HALF_EXTENTS[zone]
            bounds = (
                x - half_x - footprint_radius_m,
                x + half_x + footprint_radius_m,
                y - half_y - footprint_radius_m,
                y + half_y + footprint_radius_m,
            )
            if self._segment_intersects_box(start, end, bounds):
                return False
        return True

    @staticmethod
    def _segment_intersects_box(
        start: "Pose",
        end: "Pose",
        bounds: Tuple[float, float, float, float],
    ) -> bool:
        """Return whether a segment touches an axis-aligned closed box."""
        t_min, t_max = 0.0, 1.0
        for origin, delta, lower, upper in (
            (start.x, end.x - start.x, bounds[0], bounds[1]),
            (start.y, end.y - start.y, bounds[2], bounds[3]),
        ):
            if abs(delta) <= 1e-12:
                if origin < lower or origin > upper:
                    return False
                continue
            entry = (lower - origin) / delta
            exit_ = (upper - origin) / delta
            if entry > exit_:
                entry, exit_ = exit_, entry
            t_min = max(t_min, entry)
            t_max = min(t_max, exit_)
            if t_min > t_max:
                return False
        return True

    def get_access_point(
        self,
        target_coords: Tuple[int, int],
        robot_coords: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Return the nearest chassis cell from which the arm can reach a target."""
        return self.get_access_points(target_coords, robot_coords)[0]

    def get_access_points(
        self,
        target_coords: Tuple[int, int],
        robot_coords: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], ...]:
        """Return every valid access cell, nearest first and deterministic."""
        x, y = target_coords
        if self.get_zone_type(x, y) not in self.MANIPULATION_ZONES:
            if not self.is_navigable(x, y):
                raise ValueError(f"Target {target_coords} is not navigable")
            return (target_coords,)

        candidates = tuple(
            (x + dx, y + dy)
            for dx, dy in self._ACCESS_OFFSETS
            if self.is_navigable(x + dx, y + dy)
        )
        if not candidates:
            raise ValueError(
                f"Target {target_coords} has no navigable access point"
            )

        robot_x, robot_y = robot_coords
        return tuple(
            sorted(
                candidates,
                key=lambda point: (
                    abs(point[0] - robot_x) + abs(point[1] - robot_y),
                    point,
                ),
            )
        )

    def get_approach_direction(
        self,
        target_coords: Tuple[int, int],
        access_coords: Tuple[int, int],
    ) -> CardinalDirection:
        """Return the world direction from an access cell toward its target."""
        delta = (
            target_coords[0] - access_coords[0],
            target_coords[1] - access_coords[1],
        )
        directions = {
            (1, 0): CardinalDirection.POS_X,
            (-1, 0): CardinalDirection.NEG_X,
            (0, 1): CardinalDirection.POS_Y,
            (0, -1): CardinalDirection.NEG_Y,
        }
        try:
            return directions[delta]
        except KeyError as exc:
            raise ValueError(
                f"Access cell {access_coords} must be cardinally adjacent "
                f"to target {target_coords}"
            ) from exc
