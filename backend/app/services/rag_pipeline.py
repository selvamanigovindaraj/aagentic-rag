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

_SYNTHESIS_ROUTES = {Route.COMPARISON, Route.TEMPORAL, Route.MULTIHOP}
_PRO_ROUTES = {Route.MULTIHOP, Route.TEMPORAL}
_OUT_OF_SCOPE_ANSWER = "I can only answer questions grounded in the authorized document corpus."
_NO_EVIDENCE_ANSWER = "I could not find enough authorized evidence to answer reliably."


def _log_fallback(node: str, exc: Exception) -> None:
    # Malformed structured output degrades silently by design (never retries the
    # durable run) — this is the only signal an operator has that it happened.
    logger.warning("fallback_used", extra={"node": node, "reason": type(exc).__name__})


def _question(state: AgentState) -> str:
    return f"<question>{state['query']}</question>"


def _auth_from_state(state: AgentState) -> AuthContext:
    return AuthContext(
        subject=state["user_id"],
        tenant_id=state["tenant_id"],
        groups=frozenset(state["groups"]),
    )


def _keyword_plan(query: str, route: Route) -> ReasoningPlan:
    return ReasoningPlan.model_validate(
        {
            "known_entities": [],
            "unknown_entities": [],
            "leaves": [
                {"id": f"leaf_{index}", "question": question, "depends_on": []}
                for index, question in enumerate(decompose(query, route))
            ],
        }
    )


def _initial_leaves(plan: ReasoningPlan, route: Route, expanded_query: str | None) -> list[dict]:
    return [
        {
            **leaf.model_dump(),
            "query": (
                expanded_query or leaf.question
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


def _sources_block(state: AgentState, evidence: list[Evidence]) -> str:
    by_id = {item.id: item for item in evidence}
    groups = state.get("context_groups") or [
        {"title": item.document_title, "ids": [item.id]} for item in evidence
    ]
    blocks = []
    for group in groups:
        seen_contexts: set[str] = set()
        snippets = []
        for item_id in group["ids"]:
            context = by_id[item_id].context_text or by_id[item_id].text
            if context in seen_contexts:
                continue
            seen_contexts.add(context)
            snippets.append(f"ID: {item_id}\n{context}")
        blocks.append(f"Source: {group['title']}\n" + "\n".join(snippets))
    return "\n\n".join(blocks)


def _claims_fully_cited(grounded: GroundedAnswer, sources: dict[str, Evidence]) -> bool:
    return all(
        claim.evidence_ids and set(claim.evidence_ids) <= sources.keys()
        for claim in grounded.claims
    )


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

    async def _structured(self, node: str, prompt_key: str, user_content: str, schema):
        """One structured LLM call; malformed output returns None instead of raising."""
        try:
            raw = await self.models.complete(
                [
                    {"role": "system", "content": prompt(prompt_key)},
                    {"role": "user", "content": user_content},
                ],
                json_output=True,
            )
            return schema.model_validate_json(raw)
        except (ValueError, json.JSONDecodeError, OpenAIError) as exc:
            _log_fallback(node, exc)
            return None

    def _has_model_budget(self, model_calls: int) -> bool:
        return bool(self.models) and model_calls < self.settings.max_total_model_calls

    async def _classify(self, state: AgentState) -> dict:
        route = classify(state["query"])
        model_calls = state.get("model_calls", 0)
        expansion = None
        if self.models:
            # The keyword router's fallback for anything it doesn't recognize is
            # DIRECT -- that's exactly the case that most needs the LLM's second
            # look, not a case to skip it for (real-world phrasing rarely uses the
            # router's literal trigger words for comparison/temporal/multi-hop).
            expansion = await self._structured(
                "classify", "router-expansion:v1", _question(state), RoutingExpansion
            )
            route = Route(expansion.route) if expansion else route
            model_calls += 1
        if route == Route.OUT_OF_SCOPE:
            return {
                "route": route,
                "status": "complete",
                "answer": _OUT_OF_SCOPE_ANSWER,
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
        if not self._has_model_budget(state.get("model_calls", 0)):
            return {"expanded_query": state["query"]}
        expansion = await self._structured(
            "expand", "query-expansion:v1", _question(state), QueryExpansion
        )
        return {
            "expanded_query": expansion.expanded_query if expansion else state["query"],
            "expansion_terms": expansion.terms if expansion else [],
            "model_calls": state.get("model_calls", 0) + 1,
        }

    async def _plan(self, state: AgentState) -> dict:
        route = Route(state["route"])
        plan = _keyword_plan(state["query"], route)
        model_calls = state.get("model_calls", 0)
        if route in _SYNTHESIS_ROUTES and self._has_model_budget(model_calls):
            llm_plan = await self._structured(
                "plan", "reasoning-plan:v1", _question(state), ReasoningPlan
            )
            plan = llm_plan or plan
            model_calls += 1
        return {
            "reasoning_plan": plan.model_dump(),
            "leaf_states": _initial_leaves(plan, route, state.get("expanded_query")),
            "subquestions": [leaf.question for leaf in plan.leaves],
            "accepted_evidence": [],
            "needs_escalation": False,
            "model_calls": model_calls,
            "status": "searching",
        }

    async def _retrieve(self, state: AgentState) -> dict:
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
        await self._fill_candidates(selected, Route(state["route"]), _auth_from_state(state))
        return {
            "leaf_states": leaves,
            "attempts": attempts,
            "retrieval_calls": retrieval_calls + len(selected),
            "status": "verifying",
        }

    async def _fill_candidates(self, leaves: list[dict], route: Route, auth: AuthContext) -> None:
        results = await asyncio.gather(
            *(
                self.retriever.retrieve(
                    leaf["query"], route, auth, self.settings.max_retrieval_candidates
                )
                for leaf in leaves
            )
        )
        for leaf, items in zip(leaves, results, strict=True):
            leaf["candidates"] = [item.model_dump(mode="json") for item in items]
            leaf["status"] = "retrieved"

    def _authorized_leaf_items(
        self, state: AgentState, leaves: list[dict]
    ) -> dict[str, list[Evidence]]:
        return {
            leaf["id"]: authorized_evidence(
                [Evidence.model_validate(item) for item in leaf.get("candidates", [])],
                state["tenant_id"],
                frozenset(state["groups"]),
            )[: self.settings.max_evidence_per_leaf]
            for leaf in leaves
            if leaf["status"] == "retrieved"
        }

    async def _critic_decisions(
        self, state: AgentState, leaves: list[dict], leaf_items: dict[str, list[Evidence]],
        model_calls: int,
    ) -> tuple[dict, int]:
        critic_input = "\n\n".join(
            f"Leaf: {leaf['question']}\nCandidate ID: {item.id}\nText: {item.text[:2000]}"
            for leaf in leaves
            for item in leaf_items.get(leaf["id"], [])
        )
        if (
            Route(state["route"]) not in _SYNTHESIS_ROUTES
            or not critic_input
            or not self._has_model_budget(model_calls)
        ):
            return {}, model_calls
        batch = await self._structured(
            "critique", "evidence-critic:v1", critic_input, CritiqueBatch
        )
        decisions = batch.decisions if batch else []
        return {decision.evidence_id: decision for decision in decisions}, model_calls + 1

    def _grade_leaf(
        self, leaf: dict, items: list[Evidence], decision_by_id: dict, route: Route
    ) -> list[Evidence]:
        evidence_limit = (
            min(2, self.settings.max_evidence_per_leaf)
            if route == Route.DIRECT
            else self.settings.max_evidence_per_leaf
        )
        accepted = grade(leaf["question"], items, evidence_limit)
        reasons: list[str] = []
        decisions = [decision_by_id[item.id] for item in items if item.id in decision_by_id]
        if decisions:
            accepted_ids = {decision.evidence_id for decision in decisions if decision.accepted}
            accepted = [item for item in items if item.id in accepted_ids][:evidence_limit]
            reasons = [decision.reason for decision in decisions if not decision.accepted]
        leaf["accepted"] = [item.model_dump(mode="json") for item in accepted]
        leaf["rejection_reasons"] = reasons
        leaf["status"] = "accepted" if accepted else "failed"
        return accepted

    async def _critique(self, state: AgentState) -> dict:
        route = Route(state["route"])
        leaves = [dict(leaf) for leaf in state["leaf_states"]]
        leaf_items = self._authorized_leaf_items(state, leaves)
        decision_by_id, model_calls = await self._critic_decisions(
            state, leaves, leaf_items, state.get("model_calls", 0)
        )
        all_accepted: dict[str, Evidence] = {}
        for leaf in leaves:
            if leaf["status"] != "retrieved":
                for item in leaf.get("accepted", []):
                    evidence = Evidence.model_validate(item)
                    all_accepted[evidence.id] = evidence
                continue
            for item in self._grade_leaf(leaf, leaf_items[leaf["id"]], decision_by_id, route):
                all_accepted[item.id] = item
        confidence = max((item.score for item in all_accepted.values()), default=0.0)
        if route == Route.DIRECT and confidence < self.settings.direct_confidence_threshold:
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
            if self._has_model_budget(model_calls):
                expansion = await self._structured(
                    "rewrite",
                    "leaf-rewrite:v1",
                    f"Question: {leaf['question']}\n"
                    f"Rejected because: {leaf['rejection_reasons']}",
                    QueryExpansion,
                )
                query = expansion.expanded_query if expansion else query
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
                "answer": _NO_EVIDENCE_ANSWER,
                "citations": [],
                "generated_claims": 0,
                "grounded_claims": 0,
                "citation_references": 0,
                "valid_citation_references": 0,
                "answer_source": "no_evidence",
                "status": "complete",
            }
        if not self.models:
            return self._numbered_fallback(state, evidence)
        grounded = await self._grounded_answer(state, evidence)
        sources = {item.id: item for item in evidence if item.source_kind == "source"}
        if grounded and not grounded.claims:
            return self._declined_answer(state, grounded)
        if grounded and _claims_fully_cited(grounded, sources):
            return self._cited_answer(state, grounded, sources)
        if grounded:
            logger.warning(
                "fallback_used",
                extra={"node": "generate", "reason": "citation_outside_accepted_evidence"},
            )
        # The LLM was attempted but gave no verifiable answer -- refuse rather
        # than present the retrieved evidence as if it were validated.
        return self._unverifiable_answer(state)

    async def _grounded_answer(self, state: AgentState, evidence: list[Evidence]):
        # Unlike the other nodes, only the parse is guarded: a transport error
        # here must propagate to the durable run, not degrade into a fallback.
        raw = await self.models.complete(
            [
                {"role": "system", "content": prompt("grounded-claims:v1")},
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['query']}\n"
                        f"<evidence>\n{_sources_block(state, evidence)}\n</evidence>"
                    ),
                },
            ],
            use_pro=Route(state["route"]) in _PRO_ROUTES,
            json_output=True,
        )
        try:
            return GroundedAnswer.model_validate_json(raw)
        except (ValueError, json.JSONDecodeError, OpenAIError) as exc:
            _log_fallback("generate", exc)
            return None

    def _declined_answer(self, state: AgentState, grounded: GroundedAnswer) -> dict:
        # `all(...)` over an empty claims list is vacuously True -- without this
        # branch, a model correctly declining to fabricate a connection (per
        # GROUNDED_CLAIMS_V1's "put it in unsupported rather than inferring it")
        # would read as a successful, grounded, zero-content answer instead of
        # the refusal it actually is.
        answer = _NO_EVIDENCE_ANSWER
        if grounded.unsupported:
            answer += " Missing: " + "; ".join(grounded.unsupported)
        return self._generation_result(
            state, answer, used=[], claims=0, references=0, answer_source="declined"
        )

    def _cited_answer(
        self, state: AgentState, grounded: GroundedAnswer, sources: dict[str, Evidence]
    ) -> dict:
        ordered_ids = list(
            dict.fromkeys(
                evidence_id for claim in grounded.claims for evidence_id in claim.evidence_ids
            )
        )
        used = [sources[evidence_id] for evidence_id in ordered_ids]
        citation_numbers = {item.id: index for index, item in enumerate(used, 1)}
        answer = "\n\n".join(
            claim.text
            + " "
            + "".join(f"[{citation_numbers[item]}]" for item in claim.evidence_ids)
            for claim in grounded.claims
        )
        if grounded.unsupported:
            answer += "\n\nUnsupported: " + "; ".join(grounded.unsupported)
        references = sum(len(claim.evidence_ids) for claim in grounded.claims)
        return self._generation_result(
            state,
            answer,
            used,
            claims=len(grounded.claims),
            references=references,
            answer_source="grounded",
        )

    def _numbered_fallback(self, state: AgentState, evidence: list[Evidence]) -> dict:
        answer = "\n\n".join(f"{item.text} [{index}]" for index, item in enumerate(evidence, 1))
        return self._generation_result(
            state,
            answer,
            evidence,
            claims=len(evidence),
            references=len(evidence),
            answer_source="fallback",
        )

    def _unverifiable_answer(self, state: AgentState) -> dict:
        # The grounded-claims call either didn't parse or cited evidence
        # outside what was accepted -- there's no safe way to present the
        # retrieved evidence as an answer without model verification.
        return self._generation_result(
            state,
            _NO_EVIDENCE_ANSWER,
            used=[],
            claims=0,
            references=0,
            answer_source="unverifiable",
        )

    def _generation_result(
        self,
        state: AgentState,
        answer: str,
        used: list[Evidence],
        claims: int,
        references: int,
        answer_source: str,
    ) -> dict:
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
            "generated_claims": claims,
            "grounded_claims": claims,
            "citation_references": references,
            "valid_citation_references": references,
            "answer_source": answer_source,
            "model_calls": state.get("model_calls", 0) + (1 if self.models else 0),
            "status": "complete",
        }
