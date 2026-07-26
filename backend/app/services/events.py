"""Ephemeral chat progress events."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Protocol
from uuid import UUID

from redis.asyncio import Redis


class EventBroker(Protocol):
    async def publish(self, run_id: UUID, event: dict) -> None: ...
    def stream(self, run_id: UUID) -> AsyncIterator[dict]: ...


class MemoryEventBroker:
    def __init__(self) -> None:
        self.queues: dict[UUID, asyncio.Queue[dict]] = defaultdict(asyncio.Queue)

    async def publish(self, run_id: UUID, event: dict) -> None:
        await self.queues[run_id].put(event)

    async def stream(self, run_id: UUID) -> AsyncIterator[dict]:
        while True:
            event = await self.queues[run_id].get()
            yield event
            if event["type"] in {"complete", "error"}:
                return


class RedisEventBroker:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def publish(self, run_id: UUID, event: dict) -> None:
        key = f"chat:run:{run_id}"
        async with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.rpush(key, json.dumps(event))
            pipeline.expire(key, 3600)
            await pipeline.execute()

    async def stream(self, run_id: UUID) -> AsyncIterator[dict]:
        key = f"chat:run:{run_id}"
        while True:
            item = await self.redis.blpop(key, timeout=15)
            if not item:
                continue
            event = json.loads(item[1])
            yield event
            if event["type"] in {"complete", "error"}:
                return
