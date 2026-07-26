import json

from redis.asyncio import Redis
from redis.exceptions import RedisError


class SemanticCache:
    def __init__(self, redis: Redis, ttl_seconds: int = 300) -> None:
        self.redis = redis
        self.ttl_seconds = ttl_seconds

    async def get(self, key: str) -> dict | None:
        try:
            value = await self.redis.get(f"semantic:{key}")
            return json.loads(value) if value else None
        except (RedisError, json.JSONDecodeError):
            return None

    async def set(self, key: str, value: dict) -> None:
        try:
            await self.redis.set(f"semantic:{key}", json.dumps(value), ex=self.ttl_seconds)
        except RedisError:
            pass
