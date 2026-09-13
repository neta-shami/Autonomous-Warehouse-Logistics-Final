"""Events use explicit simulation time and never manufacture wall time."""

from math import nan

import pytest

from entities.enums import FaultCode
from entities.events import (
    CollisionStallEvent,
    FaultResetEvent,
    PayloadMismatchEvent,
    RelocalizationFailedEvent,
    RelocalizedEvent,
    RobotFaultedEvent,
)
from entities.pose import Pose


def test_event_requires_explicit_simulation_time():
    with pytest.raises(TypeError):
        CollisionStallEvent(
            robot_id="r1", coords=(1.0, 2.0), blocking_robot_ids=["r2"]
        )


def test_event_preserves_simulation_time_and_generates_unique_ids():
    first = CollisionStallEvent(
        sim_time=4.25,
        robot_id="r1",
        coords=(1.0, 2.0),
        blocking_robot_ids=["r2"],
    )
    second = CollisionStallEvent(
        sim_time=4.25,
        robot_id="r1",
        coords=(1.0, 2.0),
        blocking_robot_ids=["r2"],
    )

    assert first.sim_time == 4.25
    assert first.event_id != second.event_id
    assert not hasattr(first, "timestamp")


@pytest.mark.parametrize("bad_time", [-0.1, nan])
def test_event_rejects_invalid_simulation_time(bad_time):
    with pytest.raises(ValueError):
        CollisionStallEvent(
            sim_time=bad_time,
            robot_id="r1",
            coords=(1.0, 2.0),
            blocking_robot_ids=["r2"],
        )


def test_all_localization_recovery_and_health_events_are_constructible():
    events = (
        RelocalizedEvent(
            sim_time=2.4,
            robot_id="r1",
            recovered=Pose(4, 5),
            recovery_duration_s=0.4,
        ),
        RelocalizationFailedEvent(sim_time=3.0, robot_id="r1"),
        PayloadMismatchEvent(
            sim_time=5.0,
            robot_id="r1",
        ),
        RobotFaultedEvent(
            sim_time=6.0,
            robot_id="r1",
        ),
        FaultResetEvent(
            sim_time=7.0,
            robot_id="r1",
            cleared_fault=FaultCode.SENSOR_INVALID,
        ),
    )

    assert [event.sim_time for event in events] == [
        2.4,
        3.0,
        5.0,
        6.0,
        7.0,
    ]
