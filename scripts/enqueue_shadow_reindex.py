"""Queue active documents for a separately deployed target-version worker."""

import argparse
import asyncio

import asyncpg
from app.core.config import get_settings


async def main(version: str, tenant: str | None) -> None:
    settings = get_settings()
    connection = await asyncpg.connect(settings.database_url)
    try:
        count = await connection.fetchval(
            """WITH queued AS (
              INSERT INTO ingestion_jobs
                (id,tenant_id,document_id,operation,status,stage,progress,index_version,created_at)
              SELECT gen_random_uuid(),d.tenant_id,d.id,'index','queued','queued',0,$1,now()
              FROM documents d
              WHERE d.deleted_at IS NULL AND d.index_status='active'
                AND ($2::text IS NULL OR d.tenant_id=$2)
                AND NOT EXISTS (
                  SELECT 1 FROM ingestion_jobs j
                  WHERE j.document_id=d.id AND j.index_version=$1 AND j.status!='failed'
                )
              RETURNING id,tenant_id,index_version
            ), events AS (
              INSERT INTO ingestion_job_events
                (job_id,tenant_id,status,stage,progress,attempt,model_calls,index_version)
              SELECT id,tenant_id,'queued','queued',0,0,0,index_version FROM queued
            ) SELECT count(*) FROM queued""",
            version,
            tenant,
        )
        print(count)
    finally:
        await connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--tenant")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.version, arguments.tenant))
