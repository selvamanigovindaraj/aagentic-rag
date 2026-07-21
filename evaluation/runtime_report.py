from __future__ import annotations

import asyncio
import json
import os

import asyncpg

QUERY = """
SELECT route,
       count(*) AS runs,
       percentile_cont(0.95) WITHIN GROUP
         (ORDER BY (metrics->>'duration_ms')::double precision) AS p95_ms,
       sum(COALESCE((metrics->>'grounded_claims')::float,0)) /
         NULLIF(sum(COALESCE((metrics->>'generated_claims')::float,0)),0)
         AS grounded_claim_rate,
       sum(COALESCE((metrics->>'valid_citation_references')::float,0)) /
         NULLIF(sum(COALESCE((metrics->>'citation_references')::float,0)),0)
         AS citation_validity,
       avg(COALESCE((metrics->>'critic_accepted')::float,0) /
           NULLIF(COALESCE((metrics->>'critic_candidates')::float,0),0))
         AS critic_acceptance,
       avg(COALESCE((metrics->>'model_calls')::float,0)) AS model_calls,
       avg(COALESCE((metrics->>'retrieval_calls')::float,0)) AS retrieval_calls,
       avg(COALESCE((metrics->>'estimated_cost_usd')::float,0)) AS estimated_cost_usd
FROM chat_runs
WHERE status='complete' AND metrics <> '{}'::jsonb
GROUP BY route ORDER BY route
"""


async def report(database_url: str) -> list[dict]:
    connection = await asyncpg.connect(database_url)
    try:
        return [dict(row) for row in await connection.fetch(QUERY)]
    finally:
        await connection.close()


if __name__ == "__main__":
    rows = asyncio.run(
        report(os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag"))
    )
    print(json.dumps(rows, indent=2, default=str))
