from dataclasses import dataclass

from entities.enums import CardinalDirection


@dataclass
class ManipulatorOperation:
    """Mutable progress record for one robot's active manipulation."""

    is_pick: bool
    expected_payload_id: str
    tier: int
    approach: CardinalDirection
    started_at: float
    stage: str = "position"
    committed: bool = False

    def matches(
        self, is_pick: bool, expected_payload_id: str, tier: int,
        approach: CardinalDirection
    ) -> bool:
        return (
            self.is_pick is is_pick
            and self.expected_payload_id == expected_payload_id
            and self.tier == tier
            and self.approach is approach
        )
