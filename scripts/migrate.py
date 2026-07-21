import asyncio
from pathlib import Path

import asyncpg
from app.core.config import get_settings


def migration_sql() -> list[str]:
    return [path.read_text() for path in sorted(Path("backend/migrations").glob("*.sql"))]


async def migrate() -> None:
    connection = await asyncpg.connect(get_settings().database_url)
    try:
        for sql in migration_sql():
            await connection.execute(sql)
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(migrate())
