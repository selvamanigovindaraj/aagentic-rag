from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from openai import OpenAIError

from ..agents.adaptive_router import classify
from ..agents.document_grader import grade
from ..agents.query_decomposer import decompose
from ..components.llm import ModelGateway
from ..components.retrieval import Retriever
from ..core.config import Settings
from ..prompts.registry import prompt
from ..schemas.agent import (
    CritiqueBatch,
    GroundedAnswer,
    QueryExpansion,
    ReasoningPlan,
    RoutingExpansion,
)
from ..schemas.domain import AgentState, AuthContext, Citation, Evidence, Route
from ..security.content_filter import authorized_evidence
from ..security.input_guard import validate_query
from .query_rewriter import rewrite

logger = logging.getLogger(__name__)


def _log_fallback(node: str, exc: Exception) -> None:
    # Malformed structured output degrades silently by design (never retries the
    # durable run) — this is the only signal an operator has that it happened.
    logger.warning("fallback_used", extra={"node": node, "reason": type(exc).__name__})


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        settings: Settings,
        models: ModelGateway | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.retriever = retriever
        self.settings = settings
        self.models = models
        self.graph = self._build(checkpointer)

    def _build(self, checkpointer: BaseCheckpointSaver | None):
        graph = StateGraph(AgentState)
        graph.add_node("classify", self._classify)
        graph.add_node("expand", self._expand)
        graph.add_node("plan", self._plan)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("critique", self._critique)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("compose", self._compose)
        graph.add_node("generate", self._generate)
        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify", self._after_classify, {"stop": END, "continue": "expand"}
        )
        graph.add_edge("expand", "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "critique")
        graph.add_conditional_edges(
            "critique",
            self._after_critique,
            {"retry": "rewrite", "escalate": "plan", "generate": "compose"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("compose", "generate")
        graph.add_edge("generate", END)
        return graph.compile(checkpointer=checkpointer)

    async def run(self, query: str, auth: AuthContext, thread_id: str | None = None) -> AgentState:
        return await self.graph.ainvoke(
            {
                "query": validate_query(query),
                "tenant_id": auth.tenant_id,
                "user_id": auth.subject,
                "groups": sorted(auth.groups),
                "attempts": {},
                "model_calls": 0,
                "retrieval_calls": 0,
            },
            {
                "configurable": {"thread_id": thread_id or str(uuid4())},
                "metadata": {
                    "prompt_version": "v1",
                    "index_version": self.settings.index_version,
                    "content_tracing": self.settings.allow_sensitive_tracing,
                },
                "tags": ["agentic-rag", self.settings.index_version],
            },
        )

    async def _classify(self, state: AgentState) -> dict:
        route = classify(state["query"])
        model_calls = state.get("model_calls", 0)
        expansion = None
        if self.models:
            # The keyword router's fallback for anything it doesn't recognize is
            # DIRECT -- that's exactly the case that most needs the LLM's second
            # look, not a case to skip it for (real-world phrasing rarely uses the
            # router's literal trigger words for comparison/temporal/multi-hop).
            try:
                raw = await self.models.complete(
                    [
                        {"role": "system", "content": prompt("router-expansion:v1")},
                        {"role": "user", "content": f"<question>{state['query']}</question>"},
                    ],
                    json_output=True,
                )
                expansion = RoutingExpansion.model_validate_json(raw)
                route = Route(expansion.route)
            except (ValueError, json.JSONDecodeError, OpenAIError) as exc:
                _log_fallback("classify", exc)
            model_calls += 1
        if route == Route.OUT_OF_SCOPE:
            return {
                "route": route,
                "status": "complete",
                "answer": "I can only answer questions grounded in the authorized document corpus.",
                "model_calls": model_calls,
            }
        return {
            "route": route,
            "expanded_query": expansion.expanded_query if expansion else None,
            "expansion_terms": expansion.terms if expansion else [],
            "status": "planning",
            "model_calls": model_calls,
        }

    def _after_classify(self, state: AgentState) -> str:
        return "stop" if state["route"] == Route.OUT_OF_SCOPE else "continue"

    async def _expand(self, state: AgentState) -> dict:
        if state.get("expanded_query"):
            return {}
        if Route(state["route"]) == Route.DIRECT:
            return {"expanded_query": state["query"]}
        if not self.models or state.get("model_calls", 0) >= self.settings.max_total_model_calls:
            return {"expanded_query": state["query"]}
        try:
            raw = await self.models.complete(
                [
                    {"role": "system", "content": prompt("query-expansion:v1")},
                    {"role": "user", "content": f"<question>{state['query']}</question>"},
                ],
                json_output=True,
            )
            expansion = QueryExpansion.model_validate_json(raw)
        except (ValueError, json.JSONDecodeError, OpenAIError) as exc:
            _log_fallback("expand", exc)
            return {
                "expanded_query": state["query"],
                "expansion_terms": [],
                "model_calls": state.get("model_calls", 0) + 1,
            }
        return {
            "expanded_query": expansion.expanded_query,
            "expansion_terms": expansion.terms,
            "model_calls": state.get("model_calls", 0) + 1,
        }

    async def _plan(self, state: AgentState) -> dict:
        route = Route(state["route"])
        questions = decompose(state["query"], route)
        plan = ReasoningPlan.model_validate(
            {
                "known_entities": [],
                "unknown_entities": [],
                "leaves": [
                    {"id": f"leaf_{index}", "question": question, "depends_on": []}
                    for index, question in enumerate(questions)
                ],
            }
        )
        model_calls = state.get("model_calls", 0)
        if (
            self.models
            and route in {Route.COMPARISON, Route.TEMPORAL, Route.MULTIHOP}
            and model_calls < self.settings.max_total_model_calls
        ):
            try:
                raw = await self.models.complete(
                    [
                        {"role": "system", "content": prompt("reasoning-plan:v1")},
                        {"role": "user", "content": f"<question>{state['query']}</question>"},
                    ],
                    json_output=True,
                )
                plan = ReasoningPlan.model_validate_json(raw)
            except (ValueError, json.JSONDecodeError, OpenAIError) as exc:
                _log_fallback("plan", exc)
            model_calls += 1
        leaves = [
            {
                **leaf.model_dump(),
                "query": (
                    state.get("expanded_query") or leaf.question
                    if route == Route.DIRECT or len(plan.leaves) == 1
                    else leaf.question
                ),
                "attempts": 0,
                "status": "pending",
                "candidates": [],
                "accepted": [],
                "rejection_reasons": [],
            }
            for leaf in plan.leaves
        ]
        return {
            "reasoning_plan": plan.model_dump(),
            "leaf_states": leaves,
            "subquestions": [leaf.question for leaf in plan.leaves],
            "accepted_evidence": [],
            "needs_escalation": False,
            "model_calls": model_calls,
            "status": "searching",
        }

    async def _retrieve(self, state: AgentState) -> dict:
        auth = AuthContext(
            subject=state["user_id"],
            tenant_id=state["tenant_id"],
            groups=frozenset(state["groups"]),
        )
        route = Route(state["route"])
        leaves = [dict(leaf) for leaf in state["leaf_states"]]
        attempts = dict(state.get("attempts", {}))
        retrieval_calls = state.get("retrieval_calls", 0)
        pending = [leaf for leaf in leaves if leaf["status"] in {"pending", "retry"}]
        remaining = max(0, self.settings.max_total_retrieval_calls - retrieval_calls)
        selected = pending[:remaining]
        for leaf in pending[remaining:]:
            leaf["status"] = "exhausted"
        for leaf in selected:
            # Keep the public retry accounting keyed by the leaf question.  Leaf
            # ids are internal plan details and may change when a weak direct
            # lookup is escalated into a new reasoning plan.
            attempts[leaf["question"]] = leaf["attempts"] + 1
            leaf["attempts"] += 1
        results = await asyncio.gather(
            *(
                self.retriever.retrieve(
                    leaf["query"], route, auth, self.settings.max_retrieval_candidates
                )
                for leaf in selected
            )
        )
        for leaf, items in zip(selected, results, strict=True):
            leaf["candidates"] = [item.model_dump(mode="json") for item in items]
            leaf["status"] = "retrieved"
        retrieval_calls += len(selected)
        return {
            "leaf_states": leaves,
            "attempts": attempts,
            "retrieval_calls": retrieval_calls,
            "status": "verifying",
        }

    async def _critique(self, state: AgentState) -> dict:
        leaves = [dict(leaf) for leaf in state["leaf_states"]]
        model_calls = state.get("model_calls", 0)
        all_accepted: dict[str, Evidence] = {}
        leaf_items: dict[str, list[Evidence]] = {
            leaf["id"]: authorized_evidence(
                [Evidence.model_validate(item) for item in leaf.get("candidates", [])],
                state["tenant_id"],
                frozenset(state["groups"]),
            )[: self.settings.max_evidence_per_leaf]
            for leaf in leaves
            if leaf["status"] == "retrieved"
        }
        decision_by_id = {}
        critic_input = "\n\n".join(
            f"Leaf: {leaf['question']}\nCandidate ID: {item.id}\nText: {item.text[:2000]}"
            for leaf in leaves
            for item in leaf_items.get(leaf["id"], [])
        )
        if (
            self.models
            and Route(state["route"]) in {Route.COMPARISON, Route.TEMPORAL, Route.MULTIHOP}
            and critic_input
            and model_calls < self.settings.max_total_model_calls
        ):
            try:
                raw = await self.models.complete(
                    [
                        {"role": "system", "content": prompt("evidence-critic:v1")},
                        {"role": "user", "content": critic_input},
                    ],
                    json_output=True,
                )
                decision_by_id = {
                    decision.evidence_id: decision
                    for decision in CritiqueBatch.model_validate_json(raw).decisions
                }
            except (ValueError, json.JSONDecodeError, OpenAIError) as exc:
                _log_fallback("critique", exc)
                decision_by_id = {}
            model_calls += 1
        for leaf in leaves:
            if leaf["status"] != "retrieved":
                for item in leaf.get("accepted", []):
                    evidence = Evidence.model_validate(item)
                    all_accepted[evidence.id] = evidence
                continue
            items = leaf_items[leaf["id"]]
            evidence_limit = (
                min(2, self.settings.max_evidence_per_leaf)
                if Route(state["route"]) == Route.DIRECT
                else self.settings.max_evidence_per_leaf
            )
            accepted = grade(leaf["question"], items, evidence_limit)
            reasons: list[str] = []
            decisions = [decision_by_id[item.id] for item in items if item.id in decision_by_id]
            if decisions:
                accepted_ids = {decision.evidence_id for decision in decisions if decision.accepted}
                accepted = [item for item in items if item.id in accepted_ids][
                    :evidence_limit
                ]
                reasons = [decision.reason for decision in decisions if not decision.accepted]
            leaf["accepted"] = [item.model_dump(mode="json") for item in accepted]
            leaf["rejection_reasons"] = reasons
            leaf["status"] = "accepted" if accepted else "failed"
            for item in accepted:
                all_accepted[item.id] = item
        confidence = max((item.score for item in all_accepted.values()), default=0.0)
        if (
            Route(state["route"]) == Route.DIRECT
            and confidence < self.settings.direct_confidence_threshold
        ):
            return {
                "route": Route.MULTIHOP,
                "escalated": True,
                "needs_escalation": True,
                "leaf_states": [],
                "accepted_evidence": [],
                "confidence": confidence,
                "model_calls": model_calls,
            }
        return {
            "leaf_states": leaves,
            "accepted_evidence": [item.model_dump(mode="json") for item in all_accepted.values()],
            "confidence": confidence,
            "critic_candidates": sum(len(items) for items in leaf_items.values()),
            "critic_accepted": len(all_accepted),
            "model_calls": model_calls,
        }

    def _after_critique(self, state: AgentState) -> str:
        if state.get("needs_escalation"):
            return "escalate"
        retryable = any(
            leaf["status"] == "failed" and leaf["attempts"] <= self.settings.max_leaf_retries
            for leaf in state["leaf_states"]
        )
        return "retry" if retryable else "generate"

    async def _rewrite(self, state: AgentState) -> dict:
        leaves = [dict(leaf) for leaf in state["leaf_states"]]
        model_calls = state.get("model_calls", 0)
        for leaf in leaves:
            if leaf["status"] != "failed":
                continue
            query = rewrite(leaf["question"], leaf["attempts"] + 1)
            if self.models and model_calls < self.settings.max_total_model_calls:
                try:
                    raw = await self.models.complete(
                        [
                            {"role": "system", "content": prompt("leaf-rewrite:v1")},
                            {
                                "role": "user",
                                "content": (
                                    f"Question: {leaf['question']}\n"
                                    f"Rejected because: {leaf['rejection_reasons']}"
                                ),
                            },
                        ],
                        json_output=True,
                    )
                    query = QueryExpansion.model_validate_json(raw).expanded_query
                except (ValueError, json.JSONDecodeError, OpenAIError) as exc:
                    _log_fallback("rewrite", exc)
                model_calls += 1
            leaf["query"] = query
            leaf["status"] = "retry"
        return {"leaf_states": leaves, "model_calls": model_calls, "status": "retrying"}

    async def _compose(self, state: AgentState) -> dict:
        evidence: dict[str, dict] = {}
        for leaf in state["leaf_states"]:
            for item in leaf.get("accepted", []):
                evidence[item["id"]] = item
        groups: dict[str, dict] = {}
        for item in evidence.values():
            group = groups.setdefault(
                item["document_id"],
                {"document_id": item["document_id"], "title": item["document_title"], "ids": []},
            )
            group["ids"].append(item["id"])
        return {
            "accepted_evidence": list(evidence.values()),
            "context_groups": list(groups.values()),
            "status": "generating",
        }

    async def _generate(self, state: AgentState) -> dict:
        evidence = [Evidence.model_validate(item) for item in state.get("accepted_evidence", [])]
        if not evidence:
            return {
                "answer": "I could not find enough authorized evidence to answer reliably.",
                "citations": [],
                "generated_claims": 0,
                "grounded_claims": 0,
                "citation_references": 0,
                "valid_citation_references": 0,
                "status": "complete",
            }
        fallback = "\n\n".join(f"{item.text} [{index}]" for index, item in enumerate(evidence, 1))
        answer = fallback
        used = evidence
        generated_claims = grounded_claims = len(evidence)
        citation_references = valid_citation_references = len(evidence)
        if self.models:
            by_id = {item.id: item for item in evidence}
            groups = state.get("context_groups") or [
                {"title": item.document_title, "ids": [item.id]} for item in evidence
            ]
            source_groups = []
            for group in groups:
                seen_contexts = set()
                snippets = []
                for item_id in group["ids"]:
                    context = by_id[item_id].context_text or by_id[item_id].text
                    if context in seen_contexts:
                        continue
                    seen_contexts.add(context)
                    snippets.append(f"ID: {item_id}\n{context}")
                source_groups.append(f"Source: {group['title']}\n" + "\n".join(snippets))
            sources = "\n\n".join(source_groups)
            route = Route(state["route"])
            raw = await self.models.complete(
                [
                    {
                        "role": "system",
                        "content": prompt("grounded-claims:v1"),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {state['query']}\n<evidence>\n{sources}\n</evidence>"
                        ),
                    },
                ],
                use_pro=route in {Route.MULTIHOP, Route.TEMPORAL},
                json_output=True,
            )
            try:
                grounded = GroundedAnswer.model_validate_json(raw)
            except (ValueError, json.JSONDecodeError, OpenAIError) as exc:
                _log_fallback("generate", exc)
                grounded = None
            by_id = {item.id: item for item in evidence if item.source_kind == "source"}
            if grounded and not grounded.claims:
                # `all(...)` over an empty claims list is vacuously True -- without this
                # branch, a model correctly declining to fabricate a connection (per
                # GROUNDED_CLAIMS_V1's "put it in unsupported rather than inferring it")
                # would read as a successful, grounded, zero-content answer instead of
                # the refusal it actually is.
                generated_claims = grounded_claims = 0
                citation_references = valid_citation_references = 0
                used = []
                answer = "I could not find enough authorized evidence to answer reliably."
                if grounded.unsupported:
                    answer += " Missing: " + "; ".join(grounded.unsupported)
            elif grounded and all(
                claim.evidence_ids and set(claim.evidence_ids) <= by_id.keys()
                for claim in grounded.claims
            ):
                generated_claims = grounded_claims = len(grounded.claims)
                citation_references = valid_citation_references = sum(
                    len(claim.evidence_ids) for claim in grounded.claims
                )
                ordered_ids = list(
                    dict.fromkeys(
                        evidence_id
                        for claim in grounded.claims
                        for evidence_id in claim.evidence_ids
                    )
                )
                used = [by_id[evidence_id] for evidence_id in ordered_ids]
                citation_numbers = {item.id: index for index, item in enumerate(used, 1)}
                answer = "\n\n".join(
                    claim.text
                    + " "
                    + "".join(f"[{citation_numbers[item]}]" for item in claim.evidence_ids)
                    for claim in grounded.claims
                )
                if grounded.unsupported:
                    answer += "\n\nUnsupported: " + "; ".join(grounded.unsupported)
            else:
                if grounded:
                    logger.warning(
                        "fallback_used",
                        extra={"node": "generate", "reason": "citation_outside_accepted_evidence"},
                    )
                answer = fallback
        citations = [
            Citation(
                tenant_id=state["tenant_id"],
                document_id=item.document_id,
                document_title=item.document_title,
                excerpt=item.text,
                page=item.page,
                section=item.section,
            )
            for item in used
        ]
        return {
            "answer": answer,
            "citations": [item.model_dump(mode="json") for item in citations],
            "citation_verification": "passed" if citations else "unsupported",
            "generated_claims": generated_claims,
            "grounded_claims": grounded_claims,
            "citation_references": citation_references,
            "valid_citation_references": valid_citation_references,
            "model_calls": state.get("model_calls", 0) + (1 if self.models else 0),
            "status": "complete",
        }
