from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QueryExpansion(BaseModel):
    expanded_query: str = Field(min_length=1, max_length=2_000)
    terms: list[str] = Field(default_factory=list, max_length=20)


class RoutingExpansion(QueryExpansion):
    route: Literal[
        "direct", "synthesis", "comparison", "temporal_causal", "multi_hop", "out_of_scope"
    ]


class PlanLeaf(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]{1,50}$")
    question: str = Field(min_length=1, max_length=2_000)
    depends_on: list[str] = Field(default_factory=list, max_length=8)


class ReasoningPlan(BaseModel):
    known_entities: list[str] = Field(default_factory=list, max_length=20)
    unknown_entities: list[str] = Field(default_factory=list, max_length=20)
    leaves: list[PlanLeaf] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def dependencies_exist(self) -> ReasoningPlan:
        identifiers = {leaf.id for leaf in self.leaves}
        if len(identifiers) != len(self.leaves):
            raise ValueError("leaf identifiers must be unique")
        if any(
            dependency not in identifiers for leaf in self.leaves for dependency in leaf.depends_on
        ):
            raise ValueError("leaf dependency does not exist")
        return self


class CritiqueDecision(BaseModel):
    evidence_id: str
    accepted: bool
    reason: str = Field(min_length=1, max_length=500)


class CritiqueBatch(BaseModel):
    decisions: list[CritiqueDecision] = Field(default_factory=list, max_length=20)


class GroundedClaim(BaseModel):
    text: str = Field(min_length=1, max_length=2_000)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class GroundedAnswer(BaseModel):
    claims: list[GroundedClaim] = Field(default_factory=list, max_length=30)
    unsupported: list[str] = Field(default_factory=list, max_length=20)
