from __future__ import annotations

import json
from uuid import UUID

import asyncpg

from ..schemas.domain import ChatRun, ChatSession, Citation, CorpusRebuild, Document, IngestionJob


class PostgresStore:
    def __init__(self, pool: asyncpg.Pool, index_version: str = "v1") -> None:
        self.pool = pool
        self.index_version = index_version

    async def create_document(self, document: Document, job: IngestionJob) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                """INSERT INTO documents
                (id, tenant_id, logical_id, revision_of, title, source_url, object_key, author,
                 department, acl_groups, content_hash, version, index_status, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)""",
                document.id,
                document.tenant_id,
                document.logical_id,
                document.revision_of,
                document.title,
                document.source_url,
                document.object_key,
                document.author,
                document.department,
                list(document.acl_groups),
                document.content_hash,
                document.version,
                document.index_status,
                document.created_at,
            )
            await connection.execute(
                """INSERT INTO ingestion_jobs
                (id, tenant_id, document_id, operation, status, stage, progress, error,
                 index_version, model_calls, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
                job.id,
                job.tenant_id,
                job.document_id,
                job.operation,
                job.status,
                job.stage,
                job.progress,
                job.error,
                job.index_version,
                job.model_calls,
                job.created_at,
            )
            await connection.execute(
                """INSERT INTO ingestion_job_events
                (job_id,tenant_id,status,stage,progress,attempt,model_calls,index_version,error)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
                job.id,
                job.tenant_id,
                job.status,
                job.stage,
                job.progress,
                job.attempts,
                job.model_calls,
                job.index_version,
                job.error,
            )

    async def find_duplicate(
        self, tenant_id: str, content_hash: str, groups: frozenset[str]
    ) -> tuple[Document, IngestionJob] | None:
        row = await self.pool.fetchrow(
            """SELECT d.*, j.id AS job_id, j.status AS job_status, j.stage AS job_stage,
                      j.progress AS job_progress, j.error AS job_error,
                      j.attempts AS job_attempts, j.model_calls AS job_model_calls,
                      j.index_version AS job_index_version,
                      j.created_at AS job_created_at
            FROM documents d JOIN LATERAL (
              SELECT * FROM ingestion_jobs WHERE document_id=d.id ORDER BY created_at DESC LIMIT 1
            ) j ON true
            WHERE d.tenant_id=$1 AND d.content_hash=$2 AND d.deleted_at IS NULL
              AND (cardinality(d.acl_groups)=0 OR d.acl_groups && $3::text[])
            ORDER BY d.version DESC LIMIT 1""",
            tenant_id,
            content_hash,
            list(groups),
        )
        if not row:
            return None
        data = dict(row)
        document = Document.model_validate(
            {key: value for key, value in data.items() if not key.startswith("job_")}
        )
        job = IngestionJob.model_validate(
            {
                "id": data["job_id"],
                "tenant_id": document.tenant_id,
                "document_id": document.id,
                "status": data["job_status"],
                "stage": data["job_stage"],
                "progress": data["job_progress"],
                "error": data["job_error"],
                "attempts": data["job_attempts"],
                "model_calls": data["job_model_calls"],
                "index_version": data["job_index_version"],
                "created_at": data["job_created_at"],
            }
        )
        return document, job

    async def get_job(self, tenant_id: str, job_id: UUID) -> IngestionJob | None:
        row = await self.pool.fetchrow(
            "SELECT * FROM ingestion_jobs WHERE tenant_id=$1 AND id=$2", tenant_id, job_id
        )
        return IngestionJob.model_validate(dict(row)) if row else None

    async def get_job_by_id(self, job_id: UUID) -> IngestionJob | None:
        row = await self.pool.fetchrow("SELECT * FROM ingestion_jobs WHERE id=$1", job_id)
        return IngestionJob.model_validate(dict(row)) if row else None

    async def update_job(self, job: IngestionJob) -> None:
        await self.pool.execute(
            """UPDATE ingestion_jobs SET status=$2,stage=$3,progress=$4,error=$5,
               attempts=$6,worker_id=$7,lease_until=$8,index_version=$9,model_calls=$10
               WHERE id=$1""",
            job.id,
            job.status,
            job.stage,
            job.progress,
            job.error,
            job.attempts,
            job.worker_id,
            job.lease_until,
            job.index_version,
            job.model_calls,
        )
        await self.pool.execute(
            """INSERT INTO ingestion_job_events
            (job_id,tenant_id,status,stage,progress,attempt,model_calls,index_version,error)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
            job.id,
            job.tenant_id,
            job.status,
            job.stage,
            job.progress,
            job.attempts,
            job.model_calls,
            job.index_version,
            job.error,
        )

    async def claim_ingestion_job(
        self, worker_id: str, *, lease_seconds: int, index_version: str = "v1"
    ) -> IngestionJob | None:
        row = await self.pool.fetchrow(
            """WITH candidate AS (
              SELECT id FROM ingestion_jobs
              WHERE attempts < 3 AND index_version=$3 AND (
                status='queued' OR (status='running' AND lease_until < now())
              )
              ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
            )
            UPDATE ingestion_jobs j SET status='running', attempts=j.attempts+1,
              worker_id=$1, lease_until=now()+($2 * interval '1 second')
            FROM candidate WHERE j.id=candidate.id RETURNING j.*""",
            worker_id,
            lease_seconds,
            index_version,
        )
        return IngestionJob.model_validate(dict(row)) if row else None

    async def request_corpus_rebuild(
        self, tenant_id: str, acl_groups: set[str], index_version: str
    ) -> None:
        from ..services.ingestion import acl_cohort

        await self.pool.execute(
            """INSERT INTO corpus_rebuilds
            (tenant_id,acl_cohort,acl_groups,index_version,status,attempts,requested_at)
            VALUES ($1,$2,$3,$4,'queued',0,now())
            ON CONFLICT (tenant_id,acl_cohort,index_version) DO UPDATE
              SET acl_groups=EXCLUDED.acl_groups,status='queued',attempts=0,
                  worker_id=NULL,requested_at=now()""",
            tenant_id,
            acl_cohort(acl_groups),
            sorted(acl_groups),
            index_version,
        )

    async def claim_corpus_rebuild(
        self, worker_id: str, index_version: str
    ) -> CorpusRebuild | None:
        row = await self.pool.fetchrow(
            """WITH candidate AS (
              SELECT tenant_id,acl_cohort,index_version FROM corpus_rebuilds
              WHERE status='queued' AND attempts < 3 AND index_version=$2
                AND NOT EXISTS (
                  SELECT 1 FROM ingestion_jobs
                  WHERE status IN ('queued','running') AND index_version=$2
                )
              ORDER BY requested_at FOR UPDATE SKIP LOCKED LIMIT 1
            )
            UPDATE corpus_rebuilds c SET status='running',attempts=c.attempts+1,worker_id=$1
            FROM candidate
            WHERE c.tenant_id=candidate.tenant_id AND c.acl_cohort=candidate.acl_cohort
              AND c.index_version=candidate.index_version
            RETURNING c.*""",
            worker_id,
            index_version,
        )
        return CorpusRebuild.model_validate(dict(row)) if row else None

    async def finish_corpus_rebuild(self, task: CorpusRebuild, success: bool) -> None:
        if success:
            await self.pool.execute(
                """DELETE FROM corpus_rebuilds
                WHERE tenant_id=$1 AND acl_cohort=$2 AND index_version=$3 AND requested_at=$4""",
                task.tenant_id,
                task.acl_cohort,
                task.index_version,
                task.requested_at,
            )
        else:
            await self.pool.execute(
                """UPDATE corpus_rebuilds SET status='queued',worker_id=NULL
                WHERE tenant_id=$1 AND acl_cohort=$2 AND index_version=$3 AND requested_at=$4""",
                task.tenant_id,
                task.acl_cohort,
                task.index_version,
                task.requested_at,
            )

    async def get_document(self, document_id: UUID) -> Document | None:
        row = await self.pool.fetchrow(
            "SELECT * FROM documents WHERE id=$1 AND deleted_at IS NULL", document_id
        )
        return Document.model_validate(dict(row)) if row else None

    async def active_document_ids(
        self, tenant_id: str, groups: frozenset[str], document_ids: set[UUID]
    ) -> set[UUID]:
        if not document_ids:
            return set()
        rows = await self.pool.fetch(
            """SELECT id FROM documents WHERE tenant_id=$1 AND id=ANY($2::uuid[])
            AND deleted_at IS NULL AND index_status='active'
            AND (cardinality(acl_groups)=0 OR acl_groups && $3::text[])""",
            tenant_id,
            list(document_ids),
            list(groups),
        )
        return {row["id"] for row in rows}

    async def activate_document(self, document: Document) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            if document.revision_of:
                await connection.execute(
                    "UPDATE documents SET index_status='superseded' WHERE id=$1",
                    document.revision_of,
                )
            await connection.execute(
                "UPDATE documents SET index_status='active' WHERE id=$1", document.id
            )

    async def finalize_document_delete(self, document_id: UUID) -> None:
        await self.pool.execute(
            "UPDATE documents SET deleted_at=now() WHERE id=$1 AND index_status='deleting'",
            document_id,
        )

    async def list_documents(self, tenant_id: str, groups: frozenset[str]) -> list[Document]:
        rows = await self.pool.fetch(
            """SELECT * FROM documents WHERE tenant_id=$1 AND deleted_at IS NULL
            AND index_status != 'deleting'
            AND (cardinality(acl_groups)=0 OR acl_groups && $2::text[]) ORDER BY created_at DESC""",
            tenant_id,
            list(groups),
        )
        return [Document.model_validate(dict(row)) for row in rows]

    async def delete_document(
        self, tenant_id: str, document_id: UUID, groups: frozenset[str]
    ) -> bool:
        async with self.pool.acquire() as connection, connection.transaction():
            result = await connection.execute(
                """UPDATE documents SET index_status='deleting' WHERE tenant_id=$1 AND id=$2
            AND deleted_at IS NULL
            AND (cardinality(acl_groups)=0 OR acl_groups && $3::text[])""",
                tenant_id,
                document_id,
                list(groups),
            )
            if result == "UPDATE 1":
                job_id = await connection.fetchval(
                    """INSERT INTO ingestion_jobs
                    (id,tenant_id,document_id,operation,status,stage,progress,index_version,created_at)
                    VALUES (gen_random_uuid(),$1,$2,'delete','queued','queued',0,$3,now())
                    RETURNING id""",
                    tenant_id,
                    document_id,
                    self.index_version,
                )
                await connection.execute(
                    """INSERT INTO ingestion_job_events
                    (job_id,tenant_id,status,stage,progress,attempt,model_calls,index_version)
                    VALUES ($1,$2,'queued','queued',0,0,0,$3)""",
                    job_id,
                    tenant_id,
                    self.index_version,
                )
        return result == "UPDATE 1"

    async def create_session(self, session: ChatSession) -> None:
        await self.pool.execute(
            """INSERT INTO chat_sessions (id,tenant_id,user_id,title,created_at)
            VALUES ($1,$2,$3,$4,$5)""",
            session.id,
            session.tenant_id,
            session.user_id,
            session.title,
            session.created_at,
        )

    async def get_session(self, tenant_id: str, session_id: UUID) -> ChatSession | None:
        row = await self.pool.fetchrow(
            "SELECT * FROM chat_sessions WHERE tenant_id=$1 AND id=$2", tenant_id, session_id
        )
        return ChatSession.model_validate(dict(row)) if row else None

    async def create_run(self, run: ChatRun) -> None:
        await self._write_run(run, insert=True)

    async def get_run(self, tenant_id: str, run_id: UUID) -> ChatRun | None:
        row = await self.pool.fetchrow(
            "SELECT * FROM chat_runs WHERE tenant_id=$1 AND id=$2", tenant_id, run_id
        )
        return self._run(row) if row else None

    async def update_run(self, run: ChatRun) -> None:
        await self._write_run(run, insert=False)

    async def claim_chat_run(self, worker_id: str, *, lease_seconds: int) -> ChatRun | None:
        row = await self.pool.fetchrow(
            """
            WITH candidate AS (
              SELECT id FROM chat_runs
              WHERE attempts < 3 AND (status='queued' OR (status='running' AND lease_until < now()))
              ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
            )
            UPDATE chat_runs r SET status='running', attempts=r.attempts+1,
              worker_id=$1, lease_until=now()+make_interval(secs => $2)
            FROM candidate WHERE r.id=candidate.id RETURNING r.*
            """,
            worker_id,
            lease_seconds,
        )
        return self._run(row) if row else None

    @staticmethod
    def _run(row: asyncpg.Record) -> ChatRun:
        values = dict(row)
        if isinstance(values.get("metrics"), str):
            values["metrics"] = json.loads(values["metrics"])
        return ChatRun.model_validate(values)

    async def _write_run(self, run: ChatRun, *, insert: bool) -> None:
        values = (
            run.id,
            run.tenant_id,
            run.session_id,
            run.user_id,
            run.query,
            run.status,
            run.route,
            run.answer,
            run.citation_ids,
            run.error,
            sorted(run.groups),
            run.attempts,
            run.worker_id,
            run.lease_until,
            json.dumps(run.metrics),
            run.created_at,
        )
        if insert:
            await self.pool.execute(
                """INSERT INTO chat_runs
                (id,tenant_id,session_id,user_id,query,status,route,answer,citation_ids,error,
                 groups,attempts,worker_id,lease_until,metrics,created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)""",
                *values,
            )
        else:
            await self.pool.execute(
                """UPDATE chat_runs SET status=$3,route=$4,answer=$5,citation_ids=$6,error=$7,
                  groups=$8,attempts=$9,worker_id=$10,lease_until=$11,metrics=$12
                WHERE id=$1 AND tenant_id=$2""",
                run.id,
                run.tenant_id,
                run.status,
                run.route,
                run.answer,
                run.citation_ids,
                run.error,
                sorted(run.groups),
                run.attempts,
                run.worker_id,
                run.lease_until,
                json.dumps(run.metrics),
            )

    async def save_citations(self, citations: list[Citation]) -> None:
        await self.pool.executemany(
            """INSERT INTO citations
            (id,tenant_id,document_id,document_title,excerpt,page,section)
            VALUES ($1,$2,$3,$4,$5,$6,$7)""",
            [
                (c.id, c.tenant_id, c.document_id, c.document_title, c.excerpt, c.page, c.section)
                for c in citations
            ],
        )

    async def get_citation(
        self, tenant_id: str, citation_id: UUID, groups: frozenset[str]
    ) -> Citation | None:
        row = await self.pool.fetchrow(
            """SELECT c.* FROM citations c JOIN documents d ON d.id=c.document_id
            WHERE c.tenant_id=$1 AND c.id=$2 AND d.deleted_at IS NULL
            AND d.index_status='active'
            AND (cardinality(d.acl_groups)=0 OR d.acl_groups && $3::text[])""",
            tenant_id,
            citation_id,
            list(groups),
        )
        return Citation.model_validate(dict(row)) if row else None

    async def save_feedback(self, tenant_id: str, run_id: UUID, payload: dict) -> bool:
        result = await self.pool.execute(
            """INSERT INTO run_feedback (run_id,tenant_id,grounded,useful,comment)
            SELECT id,tenant_id,$3,$4,$5 FROM chat_runs WHERE id=$1 AND tenant_id=$2
            ON CONFLICT (run_id) DO UPDATE SET grounded=$3,useful=$4,comment=$5""",
            run_id,
            tenant_id,
            payload["grounded"],
            payload["useful"],
            payload.get("comment"),
        )
        return result in {"INSERT 0 1", "UPDATE 1"}
