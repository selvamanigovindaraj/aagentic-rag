from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Protocol
from uuid import UUID

from ..schemas.domain import (
    ChatRun,
    ChatSession,
    Citation,
    CorpusRebuild,
    Document,
    IndexStatus,
    IngestionJob,
    JobStatus,
)


class Store(Protocol):
    async def create_document(self, document: Document, job: IngestionJob) -> None: ...
    async def find_duplicate(
        self, tenant_id: str, content_hash: str, groups: frozenset[str]
    ) -> tuple[Document, IngestionJob] | None: ...
    async def get_job(self, tenant_id: str, job_id: UUID) -> IngestionJob | None: ...
    async def get_job_by_id(self, job_id: UUID) -> IngestionJob | None: ...
    async def update_job(self, job: IngestionJob) -> None: ...
    async def claim_ingestion_job(
        self, worker_id: str, *, lease_seconds: int, index_version: str = "v1"
    ) -> IngestionJob | None: ...
    async def request_corpus_rebuild(
        self, tenant_id: str, acl_groups: set[str], index_version: str
    ) -> None: ...
    async def claim_corpus_rebuild(
        self, worker_id: str, index_version: str
    ) -> CorpusRebuild | None: ...
    async def finish_corpus_rebuild(self, task: CorpusRebuild, success: bool) -> None: ...
    async def get_document(self, document_id: UUID) -> Document | None: ...
    async def active_document_ids(
        self, tenant_id: str, groups: frozenset[str], document_ids: set[UUID]
    ) -> set[UUID]: ...
    async def activate_document(self, document: Document) -> None: ...
    async def finalize_document_delete(self, document_id: UUID) -> None: ...
    async def list_documents(self, tenant_id: str, groups: frozenset[str]) -> list[Document]: ...
    async def delete_document(
        self, tenant_id: str, document_id: UUID, groups: frozenset[str], requester_id: str
    ) -> bool: ...
    async def create_session(self, session: ChatSession) -> None: ...
    async def get_session(self, tenant_id: str, session_id: UUID) -> ChatSession | None: ...
    async def create_run(self, run: ChatRun) -> None: ...
    async def get_run(self, tenant_id: str, run_id: UUID) -> ChatRun | None: ...
    async def update_run(self, run: ChatRun) -> None: ...
    async def claim_chat_run(self, worker_id: str, *, lease_seconds: int) -> ChatRun | None: ...
    async def save_citations(self, citations: list[Citation]) -> None: ...
    async def get_citation(
        self, tenant_id: str, citation_id: UUID, groups: frozenset[str]
    ) -> Citation | None: ...
    async def save_feedback(self, tenant_id: str, run_id: UUID, payload: dict) -> bool: ...


def visible(acl: set[str], groups: frozenset[str]) -> bool:
    return not acl or bool(acl & groups)


class MemoryStore:
    def __init__(self) -> None:
        self.documents: dict[UUID, Document] = {}
        self.jobs: dict[UUID, IngestionJob] = {}
        self.sessions: dict[UUID, ChatSession] = {}
        self.runs: dict[UUID, ChatRun] = {}
        self.citations: dict[UUID, Citation] = {}
        self.feedback: dict[UUID, dict] = {}
        self.corpus_rebuilds: dict[tuple[str, str, str], CorpusRebuild] = {}
        self.events: dict[UUID, asyncio.Queue[dict]] = defaultdict(asyncio.Queue)

    async def create_document(self, document: Document, job: IngestionJob) -> None:
        self.documents[document.id] = document
        self.jobs[job.id] = job

    async def find_duplicate(
        self, tenant_id: str, content_hash: str, groups: frozenset[str]
    ) -> tuple[Document, IngestionJob] | None:
        for document in self.documents.values():
            if (
                document.tenant_id == tenant_id
                and document.content_hash == content_hash
                and visible(document.acl_groups, groups)
            ):
                job = next(
                    (job for job in self.jobs.values() if job.document_id == document.id), None
                )
                if job:
                    return document, job
        return None

    async def get_job(self, tenant_id: str, job_id: UUID) -> IngestionJob | None:
        job = self.jobs.get(job_id)
        return job if job and job.tenant_id == tenant_id else None

    async def get_job_by_id(self, job_id: UUID) -> IngestionJob | None:
        return self.jobs.get(job_id)

    async def update_job(self, job: IngestionJob) -> None:
        self.jobs[job.id] = job

    async def claim_ingestion_job(
        self, worker_id: str, *, lease_seconds: int, index_version: str = "v1"
    ) -> IngestionJob | None:
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        job = next(
            (
                item
                for item in self.jobs.values()
                if item.attempts < 3
                and item.index_version == index_version
                and (
                    item.status == "queued"
                    or (
                        item.status == "running"
                        and item.lease_until is not None
                        and item.lease_until < now
                    )
                )
            ),
            None,
        )
        if not job:
            return None
        job.status = JobStatus.RUNNING
        job.attempts += 1
        job.worker_id = worker_id
        job.lease_until = now + timedelta(seconds=lease_seconds)
        return job

    async def request_corpus_rebuild(
        self, tenant_id: str, acl_groups: set[str], index_version: str
    ) -> None:
        from ..services.ingestion import acl_cohort

        task = CorpusRebuild(
            tenant_id=tenant_id,
            acl_cohort=acl_cohort(acl_groups),
            acl_groups=acl_groups,
            index_version=index_version,
        )
        self.corpus_rebuilds[(tenant_id, task.acl_cohort, index_version)] = task

    async def claim_corpus_rebuild(
        self, worker_id: str, index_version: str
    ) -> CorpusRebuild | None:
        if any(job.status in {JobStatus.QUEUED, JobStatus.RUNNING} for job in self.jobs.values()):
            return None
        task = next(
            (
                item
                for item in self.corpus_rebuilds.values()
                if item.index_version == index_version and item.status == "queued"
            ),
            None,
        )
        if task:
            task.status, task.worker_id, task.attempts = "running", worker_id, task.attempts + 1
        return task

    async def finish_corpus_rebuild(self, task: CorpusRebuild, success: bool) -> None:
        key = (task.tenant_id, task.acl_cohort, task.index_version)
        if success:
            self.corpus_rebuilds.pop(key, None)
        else:
            task.status, task.worker_id = "queued", None

    async def get_document(self, document_id: UUID) -> Document | None:
        return self.documents.get(document_id)

    async def active_document_ids(
        self, tenant_id: str, groups: frozenset[str], document_ids: set[UUID]
    ) -> set[UUID]:
        return {
            document.id
            for document in self.documents.values()
            if document.id in document_ids
            and document.tenant_id == tenant_id
            and document.index_status == IndexStatus.ACTIVE
            and visible(document.acl_groups, groups)
        }

    async def activate_document(self, document: Document) -> None:
        if document.revision_of and document.revision_of in self.documents:
            self.documents[document.revision_of].index_status = IndexStatus.SUPERSEDED
        document.index_status = IndexStatus.ACTIVE
        self.documents[document.id] = document

    async def finalize_document_delete(self, document_id: UUID) -> None:
        self.documents.pop(document_id, None)

    async def list_documents(self, tenant_id: str, groups: frozenset[str]) -> list[Document]:
        return [
            d
            for d in self.documents.values()
            if d.tenant_id == tenant_id
            and visible(d.acl_groups, groups)
            and d.index_status != IndexStatus.DELETING
        ]

    async def delete_document(
        self, tenant_id: str, document_id: UUID, groups: frozenset[str], requester_id: str
    ) -> bool:
        document = self.documents.get(document_id)
        if (
            not document
            or document.tenant_id != tenant_id
            or not visible(document.acl_groups, groups)
            or document.uploaded_by != requester_id
        ):
            return False
        document.index_status = IndexStatus.DELETING
        cleanup = IngestionJob(tenant_id=tenant_id, document_id=document_id, operation="delete")
        self.jobs[cleanup.id] = cleanup
        return True

    async def create_session(self, session: ChatSession) -> None:
        self.sessions[session.id] = session

    async def get_session(self, tenant_id: str, session_id: UUID) -> ChatSession | None:
        session = self.sessions.get(session_id)
        return session if session and session.tenant_id == tenant_id else None

    async def create_run(self, run: ChatRun) -> None:
        self.runs[run.id] = run

    async def get_run(self, tenant_id: str, run_id: UUID) -> ChatRun | None:
        run = self.runs.get(run_id)
        return run if run and run.tenant_id == tenant_id else None

    async def update_run(self, run: ChatRun) -> None:
        self.runs[run.id] = run

    async def claim_chat_run(self, worker_id: str, *, lease_seconds: int) -> ChatRun | None:
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        run = next(
            (
                item
                for item in self.runs.values()
                if item.attempts < 3
                and (
                    item.status == "queued"
                    or (item.status == "running" and item.lease_until and item.lease_until < now)
                )
            ),
            None,
        )
        if not run:
            return None
        run.status = "running"
        run.attempts += 1
        run.worker_id = worker_id
        run.lease_until = now + timedelta(seconds=lease_seconds)
        return run

    async def save_citations(self, citations: list[Citation]) -> None:
        self.citations.update({citation.id: citation for citation in citations})

    async def get_citation(
        self, tenant_id: str, citation_id: UUID, groups: frozenset[str]
    ) -> Citation | None:
        citation = self.citations.get(citation_id)
        document = self.documents.get(citation.document_id) if citation else None
        return (
            citation
            if citation
            and citation.tenant_id == tenant_id
            and document
            and document.index_status == IndexStatus.ACTIVE
            and visible(document.acl_groups, groups)
            else None
        )

    async def save_feedback(self, tenant_id: str, run_id: UUID, payload: dict) -> bool:
        run = await self.get_run(tenant_id, run_id)
        if not run:
            return False
        self.feedback[run_id] = payload
        return True
