"""Shared validation helpers for immutable domain measurements."""

from math import isfinite
from typing import Optional, Tuple


def require_finite(
    name: str, value: float, minimum: Optional[float] = None
) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        finite = isfinite(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not finite:
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def validate_optional_velocity(
    velocity: Optional[Tuple[float, float]],
) -> None:
    if velocity is None:
        return
    if not isinstance(velocity, tuple) or len(velocity) != 2:
        raise TypeError("observed chassis velocity must be an immutable x/y tuple")
    require_finite("observed chassis vx", velocity[0])
    require_finite("observed chassis vy", velocity[1])


def validate_optional_bool(name: str, value: Optional[bool]) -> None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"{name} must be bool or None")


def require_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
