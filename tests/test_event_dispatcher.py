import unittest
from unittest.mock import MagicMock

from entities.events import CollisionStallEvent, WarehouseEvent
from use_cases.event_dispatcher import EventDispatcher


class DummyEvent(WarehouseEvent):
    pass

class TestEventDispatcher(unittest.TestCase):
    def setUp(self):
        self.dispatcher = EventDispatcher()

    def test_subscribe_and_dispatch(self):
        callback1 = MagicMock()
        callback2 = MagicMock()

        self.dispatcher.subscribe(DummyEvent, callback1)
        self.dispatcher.subscribe(DummyEvent, callback2)

        event = DummyEvent(sim_time=1.0)
        self.dispatcher.dispatch(event)

        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)

    def test_dispatch_no_subscribers_does_not_crash(self):
        event = DummyEvent(sim_time=1.0)
        try:
            self.dispatcher.dispatch(event)
        except Exception as e:
            self.fail(f"Dispatch raised {e} unexpectedly!")

    def test_dispatch_only_calls_relevant_subscribers(self):
        dummy_callback = MagicMock()
        stall_callback = MagicMock()

        self.dispatcher.subscribe(DummyEvent, dummy_callback)
        self.dispatcher.subscribe(CollisionStallEvent, stall_callback)

        event = CollisionStallEvent(
            sim_time=1.0,
            robot_id="r1",
            coords=(0, 0),
            blocking_robot_ids=["r2"],
        )
        self.dispatcher.dispatch(event)

        stall_callback.assert_called_once_with(event)
        dummy_callback.assert_not_called()

if __name__ == '__main__':
    unittest.main()
