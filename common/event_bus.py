import asyncio
from typing import Callable, Dict, List, Any, Awaitable
import logging

logger = logging.getLogger("face_ar_system.event_bus")

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Any], Awaitable[None]]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str, callback: Callable[[Any], Awaitable[None]]):
        async with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            if callback not in self._subscribers[topic]:
                self._subscribers[topic].append(callback)
                logger.debug(f"Subscribed to topic: {topic}")

    async def unsubscribe(self, topic: str, callback: Callable[[Any], Awaitable[None]]):
        async with self._lock:
            if topic in self._subscribers:
                if callback in self._subscribers[topic]:
                    self._subscribers[topic].remove(callback)
                    logger.debug(f"Unsubscribed from topic: {topic}")

    async def publish(self, topic: str, event_data: Any):
        async with self._lock:
            callbacks = list(self._subscribers.get(topic, []))
        
        if not callbacks:
            return
            
        async def safe_callback(cb, data):
            try:
                await cb(data)
            except Exception as e:
                logger.error(f"Error executing callback on topic {topic}: {e}", exc_info=True)

        # Run callbacks concurrently
        tasks = [asyncio.create_task(safe_callback(cb, event_data)) for cb in callbacks]
        await asyncio.gather(*tasks)
