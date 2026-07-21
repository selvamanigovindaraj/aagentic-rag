import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import StreamingResponse
from redis.exceptions import RedisError
from starlette.datastructures import UploadFile

from ..core.errors import AppError
from ..repositories.store import MemoryStore, Store, visible
from ..schemas.domain import (
    AuthContext,
    ChatRun,
    ChatSession,
    Citation,
    Document,
    DocumentAccepted,
    DocumentCreate,
    FeedbackCreate,
    IndexStatus,
    IngestionJob,
    MessageCreate,
    RunAccepted,
    SessionCreate,
)
from ..security.auth import auth_context

router = APIRouter(prefix="/api/v1")


def get_store(request: Request) -> Store:
    return request.app.state.store


@router.post("/documents", response_model=DocumentAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_document(
    request: Request,
    auth: AuthContext = Depends(auth_context),
    store: Store = Depends(get_store),
):
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise AppError(422, "FILE_REQUIRED", "A document file is required")
        content = await upload.read()
        if not content:
            raise AppError(422, "EMPTY_DOCUMENT", "The uploaded document is empty")
        if len(content) > request.app.state.settings.max_upload_bytes:
            raise AppError(413, "DOCUMENT_TOO_LARGE", "The uploaded document exceeds the limit")
        title = str(form.get("title") or upload.filename or "Document")
        groups = {
            item.strip() for item in str(form.get("acl_groups") or "").split(",") if item.strip()
        }
        content_hash = hashlib.sha256(content).hexdigest()
        revision_value = form.get("revision_of")
        try:
            revision_of = UUID(str(revision_value)) if revision_value else None
        except ValueError as exc:
            raise AppError(422, "INVALID_REVISION", "revision_of must be a UUID") from exc
        if not revision_of:
            duplicate = await store.find_duplicate(auth.tenant_id, content_hash, auth.groups)
            if duplicate:
                document, job = duplicate
                return DocumentAccepted(document=document, ingestion_job=job)
        document_id = uuid4()
        suffix = Path(upload.filename or "document").suffix.lower()
        object_key = f"{auth.tenant_id}/{document_id}{suffix}"
        object_store = getattr(request.app.state, "object_store", None)
        if object_store:
            await object_store.put(object_key, content, upload.content_type)
        else:
            path = Path(request.app.state.settings.object_storage_path) / object_key
            path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(path.write_bytes, content)
        payload = DocumentCreate(
            title=title,
            object_key=object_key,
            acl_groups=groups,
            content_hash=content_hash,
            revision_of=revision_of,
        )
    else:
        payload = DocumentCreate.model_validate(await request.json())
    if not payload.object_key:
        raise AppError(
            422,
            "OBJECT_REFERENCE_REQUIRED",
            "object_key is required; source_url is metadata and is not fetched",
        )
    if payload.content_hash and not payload.revision_of:
        duplicate = await store.find_duplicate(auth.tenant_id, payload.content_hash, auth.groups)
        if duplicate:
            document, job = duplicate
            return DocumentAccepted(document=document, ingestion_job=job)
    version = 1
    logical_id = None
    if payload.revision_of:
        base = await store.get_document(payload.revision_of)
        if (
            not base
            or base.tenant_id != auth.tenant_id
            or not visible(base.acl_groups, auth.groups)
        ):
            raise AppError(404, "DOCUMENT_NOT_FOUND", "Revision source not found")
        logical_id = base.logical_id
        version = base.version + 1
    document = Document(
        tenant_id=auth.tenant_id,
        logical_id=logical_id,
        version=version,
        index_status=IndexStatus.PENDING,
        **payload.model_dump(mode="json"),
    )
    job = IngestionJob(
        tenant_id=auth.tenant_id,
        document_id=document.id,
        index_version=request.app.state.settings.index_version,
    )
    await store.create_document(document, job)
    if hasattr(request.app.state, "ingestion_jobs"):
        request.app.state.ingestion_jobs.append(str(job.id))
    elif hasattr(request.app.state, "redis"):
        await request.app.state.redis.rpush("ingestion:jobs", str(job.id))
    return DocumentAccepted(document=document, ingestion_job=job)


@router.get("/ingestion-jobs/{job_id}", response_model=IngestionJob)
async def get_job(
    job_id: UUID, auth: AuthContext = Depends(auth_context), store: Store = Depends(get_store)
):
    job = await store.get_job(auth.tenant_id, job_id)
    if not job:
        raise AppError(404, "JOB_NOT_FOUND", "Ingestion job not found")
    return job


@router.get("/documents", response_model=list[Document])
async def list_documents(
    auth: AuthContext = Depends(auth_context), store: Store = Depends(get_store)
):
    return await store.list_documents(auth.tenant_id, auth.groups)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID, auth: AuthContext = Depends(auth_context), store: Store = Depends(get_store)
):
    if not await store.delete_document(auth.tenant_id, document_id, auth.groups):
        raise AppError(404, "DOCUMENT_NOT_FOUND", "Document not found")
    return Response(status_code=204)


@router.post("/chat/sessions", response_model=ChatSession, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreate,
    auth: AuthContext = Depends(auth_context),
    store: Store = Depends(get_store),
):
    session = ChatSession(tenant_id=auth.tenant_id, user_id=auth.subject, title=payload.title)
    await store.create_session(session)
    return session


@router.post(
    "/chat/sessions/{session_id}/messages",
    response_model=RunAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_message(
    session_id: UUID,
    payload: MessageCreate,
    request: Request,
    auth: AuthContext = Depends(auth_context),
    store: Store = Depends(get_store),
):
    session = await store.get_session(auth.tenant_id, session_id)
    if not session or session.user_id != auth.subject:
        raise AppError(404, "SESSION_NOT_FOUND", "Chat session not found")
    run = ChatRun(
        tenant_id=auth.tenant_id,
        session_id=session.id,
        user_id=auth.subject,
        query=payload.content,
        groups=set(auth.groups),
    )
    await store.create_run(run)
    if isinstance(store, MemoryStore):
        # Hermetic adapter: execute synchronously because there is deliberately
        # no external worker in unit/API tests. Production always uses leases.
        result = await request.app.state.agent.run(run.query, auth, str(run.id))
        citations = [Citation.model_validate(item) for item in result.get("citations", [])]
        await store.save_citations(citations)
        run.route = result.get("route")
        run.answer = result.get("answer")
        run.citation_ids = [item.id for item in citations]
        run.status = "complete"
        await store.update_run(run)
        return RunAccepted(run_id=run.id, events_url=f"/api/v1/chat/runs/{run.id}/events")
    try:
        await request.app.state.redis.rpush("chat:runs", str(run.id))
    except (AttributeError, RedisError):
        # PostgreSQL is the durable queue; Redis only reduces worker polling latency.
        pass
    return RunAccepted(run_id=run.id, events_url=f"/api/v1/chat/runs/{run.id}/events")


async def stream_events(events: AsyncIterator[dict]) -> AsyncIterator[str]:
    async for event in events:
        yield f"data: {json.dumps(event)}\n\n"


@router.get("/chat/runs/{run_id}/events")
async def run_events(
    run_id: UUID,
    request: Request,
    auth: AuthContext = Depends(auth_context),
    store: Store = Depends(get_store),
):
    run = await store.get_run(auth.tenant_id, run_id)
    if not run or run.user_id != auth.subject:
        raise AppError(404, "RUN_NOT_FOUND", "Chat run not found")
    if run.status == "complete":

        async def completed():
            payload = {
                "type": "answer",
                "content": run.answer,
                "citations": [str(item) for item in run.citation_ids],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield 'data: {"type": "complete"}\n\n'

        return StreamingResponse(completed(), media_type="text/event-stream")
    return StreamingResponse(
        stream_events(request.app.state.events.stream(run.id)),
        media_type="text/event-stream",
    )


@router.post("/chat/runs/{run_id}/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def save_feedback(
    run_id: UUID,
    payload: FeedbackCreate,
    auth: AuthContext = Depends(auth_context),
    store: Store = Depends(get_store),
):
    run = await store.get_run(auth.tenant_id, run_id)
    if not run or run.user_id != auth.subject:
        raise AppError(404, "RUN_NOT_FOUND", "Chat run not found")
    if not await store.save_feedback(auth.tenant_id, run_id, payload.model_dump()):
        raise AppError(404, "RUN_NOT_FOUND", "Chat run not found")
    return Response(status_code=204)


@router.get("/citations/{citation_id}", response_model=Citation)
async def get_citation(
    citation_id: UUID, auth: AuthContext = Depends(auth_context), store: Store = Depends(get_store)
):
    citation = await store.get_citation(auth.tenant_id, citation_id, auth.groups)
    if not citation:
        raise AppError(404, "CITATION_NOT_FOUND", "Citation not found")
    return citation
