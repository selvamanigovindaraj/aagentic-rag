import asyncio

import asyncpg
import httpx
from app.core.config import get_settings
from redis.asyncio import Redis


async def check() -> None:
    settings = get_settings()
    connection = await asyncpg.connect(settings.database_url)
    await connection.fetchval("SELECT 1")
    await connection.close()
    redis = Redis.from_url(settings.redis_url)
    await redis.ping()
    await redis.aclose()
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(f"{settings.weaviate_url.rstrip('/')}/v1/.well-known/ready")
        response.raise_for_status()
    print("postgres=ok redis=ok weaviate=ok")


if __name__ == "__main__":
    asyncio.run(check())
