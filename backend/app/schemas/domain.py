"""Public API and agent-state schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, model_validator


def now() -> datetime:
    return datetime.now(UTC)


class Route(StrEnum):
    DIRECT = "direct"
    SYNTHESIS = "synthesis"
    COMPARISON = "comparison"
    TEMPORAL = "temporal_causal"
    MULTIHOP = "multi_hop"
    OUT_OF_SCOPE = "out_of_scope"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class IndexStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETING = "deleting"


class AuthContext(BaseModel):
    subject: str
    tenant_id: str
    groups: frozenset[str] = frozenset()


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    source_url: HttpUrl | None = None
    object_key: str | None = Field(default=None, max_length=1000)
    author: str | None = Field(default=None, max_length=200)
    department: str | None = Field(default=None, max_length=200)
    acl_groups: set[str] = Field(default_factory=set)
    content_hash: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    revision_of: UUID | None = None


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    logical_id: UUID | None = None
    revision_of: UUID | None = None
    title: str
    source_url: str | None = None
    object_key: str | None = None
    author: str | None = None
    department: str | None = None
    acl_groups: set[str] = Field(default_factory=set)
    uploaded_by: str = ""
    content_hash: str | None = None
    version: int = 1
    index_status: IndexStatus = IndexStatus.ACTIVE
    created_at: datetime = Field(default_factory=now)

    @model_validator(mode="after")
    def set_logical_id(self) -> Document:
        if self.logical_id is None:
            self.logical_id = self.id
        return self


class IngestionJob(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    document_id: UUID
    operation: Literal["index", "delete"] = "index"
    status: JobStatus = JobStatus.QUEUED
    stage: str = "queued"
    progress: int = Field(default=0, ge=0, le=100)
    error: str | None = None
    attempts: int = 0
    model_calls: int = 0
    index_version: str = "v1"
    worker_id: str | None = None
    lease_until: datetime | None = None
    created_at: datetime = Field(default_factory=now)


class CorpusRebuild(BaseModel):
    tenant_id: str
    acl_cohort: str
    acl_groups: set[str]
    index_version: str
    status: Literal["queued", "running"] = "queued"
    attempts: int = 0
    worker_id: str | None = None
    requested_at: datetime = Field(default_factory=now)


class DocumentAccepted(BaseModel):
    document: Document
    ingestion_job: IngestionJob


class ChatSession(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    user_id: str
    title: str | None = None
    created_at: datetime = Field(default_factory=now)


class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=20_000)


class Citation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    document_id: UUID
    document_title: str
    excerpt: str
    page: int | None = None
    section: str | None = None


class Evidence(BaseModel):
    id: str
    document_id: UUID
    document_title: str
    text: str
    score: float
    page: int | None = None
    section: str | None = None
    acl_groups: set[str] = Field(default_factory=set)
    source_kind: Literal["source", "summary", "triple", "hyde", "reasoning"] = "source"
    context_text: str | None = None


class ChatRun(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    session_id: UUID
    user_id: str
    query: str
    status: Literal["queued", "running", "complete", "failed"] = "queued"
    route: Route | None = None
    answer: str | None = None
    citation_ids: list[UUID] = Field(default_factory=list)
    error: str | None = None
    groups: set[str] = Field(default_factory=set)
    attempts: int = 0
    worker_id: str | None = None
    lease_until: datetime | None = None
    metrics: dict[str, int | float | str | bool] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now)


class RunAccepted(BaseModel):
    run_id: UUID
    events_url: str


class FeedbackCreate(BaseModel):
    grounded: bool
    useful: bool
    comment: str | None = Field(default=None, max_length=2000)


class AgentState(TypedDict, total=False):
    query: str
    tenant_id: str
    user_id: str
    groups: list[str]
    route: str
    expanded_query: str
    subquestions: list[str]
    attempts: dict[str, int]
    evidence: list[dict]
    accepted_evidence: list[dict]
    context_groups: list[dict]
    confidence: float
    answer: str
    citations: list[dict]
    status: str
    reasoning_plan: dict
    leaf_states: list[dict]
    expansion_terms: list[str]
    model_calls: int
    retrieval_calls: int
    escalated: bool
    needs_escalation: bool
    citation_verification: str
    generated_claims: int
    grounded_claims: int
    citation_references: int
    valid_citation_references: int
    critic_candidates: int
    critic_accepted: int
