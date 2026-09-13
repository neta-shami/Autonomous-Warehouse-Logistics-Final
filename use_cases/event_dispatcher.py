"""
Event Dispatcher Module

Role: Provides the central Event Bus for the Event-Driven Architecture (EDA).
It is in charge of decoupling publishers (components that generate events) from
subscribers (components that react to events), allowing the system to scale
and change without tight coupling.
"""

from typing import Any, Callable, Dict, List, Type, TypeVar

from entities.events import WarehouseEvent

EventT = TypeVar("EventT", bound=WarehouseEvent)

class EventDispatcher:
    """
    [Pattern: Observer / Pub-Sub (Event Bus)]
    A centralized pub-sub event bus that routes WarehouseEvents
    to subscribed callback functions (Observers/Subscribers).
    This strictly decouples event producers from consumers to satisfy OCP.
    """
    def __init__(self):
        self._subscribers: Dict[Type[Any], List[Callable[..., Any]]] = {}

    def subscribe(
        self,
        event_type: Type[EventT],
        callback: Callable[[EventT], Any],
    ) -> None:
        """
        [EDA Orchestration]
        Subscribes a callback function (Listener) to a specific event type.
        When this event type is dispatched, the callback will be invoked.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def dispatch(self, event: WarehouseEvent) -> None:
        """
        [Event Hub]
        Dispatches an event to all subscribed callbacks synchronously.
        This is called by the Orchestrator or direct Publishers.
        """
        event_type = type(event)
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                callback(event)
