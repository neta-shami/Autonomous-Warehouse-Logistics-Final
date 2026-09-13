import uuid
from typing import List

from entities.enums import TaskType
from entities.task import Task
from entities.warehouse_topology import WarehouseTopology


class CLIParser:
    @staticmethod
    def parse_commands(
        command_strings: List[str], topology: WarehouseTopology
    ) -> List[Task]:
        """Parse commands and reject endpoints that do not exist physically."""
        if not isinstance(topology, WarehouseTopology):
            raise TypeError("topology must be a WarehouseTopology")
        parsed: List[Task] = []
        for cmd in command_strings:
            if not isinstance(cmd, str):
                print(
                    f"[CLIParser] Failed to parse command {cmd!r}: "
                    "command must be text"
                )
                continue
            parts = cmd.strip().split()
            if not parts:
                continue

            action_str = parts[0].upper()
            try:
                if action_str == "STORE":
                    pkg_id = parts[1]
                    tx, ty = map(int, parts[2].split(','))
                    tier = int(parts[3])
                    topology.validate_shelf_slot(
                        (tx, ty), tier, field_name="STORE target"
                    )
                    parsed.append(Task.transfer(
                        task_id=f"st_{pkg_id}",
                        task_type=TaskType.STORE,
                        source_coords=(9, 1),
                        source_tier=0,
                        target_coords=(tx, ty),
                        target_tier=tier,
                        payload_id=pkg_id,
                    ))
                elif action_str == "RETRIEVE":
                    pkg_id = parts[1]
                    sx, sy = map(int, parts[2].split(','))
                    tier = int(parts[3])
                    topology.validate_shelf_slot(
                        (sx, sy), tier, field_name="RETRIEVE source"
                    )
                    parsed.append(Task.transfer(
                        task_id=f"rt_{pkg_id}",
                        task_type=TaskType.RETRIEVE,
                        source_coords=(sx, sy),
                        source_tier=tier,
                        target_coords=(9, 8),
                        target_tier=0,
                        payload_id=pkg_id,
                    ))
                elif action_str == "MOVE":
                    # MOVE src_x,src_y src_tier tgt_x,tgt_y tgt_tier
                    src_x, src_y = map(int, parts[1].split(','))
                    src = (src_x, src_y)
                    src_tier = int(parts[2])
                    tgt_x, tgt_y = map(int, parts[3].split(','))
                    tgt = (tgt_x, tgt_y)
                    tgt_tier = int(parts[4])
                    topology.validate_shelf_slot(
                        src, src_tier, field_name="MOVE source"
                    )
                    topology.validate_shelf_slot(
                        tgt, tgt_tier, field_name="MOVE target"
                    )
                    parsed.append(Task.transfer(
                        task_id=f"rel_{uuid.uuid4().hex[:8]}",
                        task_type=TaskType.RELOCATE,
                        source_coords=src, source_tier=src_tier,
                        target_coords=tgt, target_tier=tgt_tier
                    ))
                else:
                    print(
                        f"[CLIParser] Unknown command '{action_str}' in '{cmd}'. "
                        f"Expected one of: STORE, RETRIEVE, MOVE."
                    )
            except (IndexError, TypeError, ValueError) as e:
                print(f"[CLIParser] Failed to parse command '{cmd}': {e}")
        return parsed
