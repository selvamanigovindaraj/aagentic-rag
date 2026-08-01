import asyncio
from uuid import uuid4

import pytest
from app.agents.adaptive_router import classify
from app.core.config import Settings
from app.schemas.domain import AuthContext, Evidence, Route
from app.services.rag_pipeline import RagPipeline


class AgenticModels:
    async def complete(self, messages, *, use_pro=False, json_output=False):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "Classify and expand" in system:
            return (
                '{"route":"multi_hop","expanded_query":"expanded investigation","terms":["bridge"]}'
            )
        if "retrieval-only vocabulary" in system:
            return '{"expanded_query":"expanded investigation","terms":["bridge"]}'
        if "complete investigation" in system:
            return (
                '{"known_entities":["A"],"unknown_entities":["bridge"],"leaves":['
                '{"id":"leaf_a","question":"Find A","depends_on":[]},'
                '{"id":"leaf_b","question":"Find bridge","depends_on":["leaf_a"]}]}'
            )
        if "Judge each candidate" in system:
            evidence_id = user.split("ID: ", 1)[1].split("\n", 1)[0]
            accepted = "supported" in user
            reason = "supports leaf" if accepted else "missing"
            return (
                f'{{"decisions":[{{"evidence_id":"{evidence_id}",'
                f'"accepted":{str(accepted).lower()},"reason":"{reason}"}}]}}'
            )
        if "Rewrite one failed" in system:
            return '{"expanded_query":"rewritten bridge query","terms":["bridge"]}'
        if "atomic factual claims" in system:
            return (
                '{"claims":[{"text":"A connects through the bridge.",'
                '"evidence_ids":["a","b"]}],"unsupported":[]}'
            )
        raise AssertionError(system)


class LeafRetriever:
    def __init__(self):
        self.calls = []

    async def retrieve(self, query, route, auth, limit):
        self.calls.append(query)
        if query == "Find A":
            return [_evidence("a", "supported A evidence")]
        if query == "rewritten bridge query":
            return [_evidence("b", "supported bridge evidence")]
        return [_evidence("rejected", "topical only")]


def _evidence(identifier: str, text: str, score: float = 0.9) -> Evidence:
    return Evidence(
        id=identifier,
        document_id=uuid4(),
        document_title=identifier,
        text=text,
        score=score,
    )


@pytest.mark.asyncio
async def test_reasoning_tree_retries_only_the_failed_leaf_with_a_rewrite():
    retriever = LeafRetriever()
    result = await RagPipeline(retriever, Settings(max_leaf_retries=2), AgenticModels()).run(
        "Find the multi-hop connection", AuthContext(subject="user", tenant_id="tenant")
    )

    assert retriever.calls.count("Find A") == 1
    assert retriever.calls.count("Find bridge") == 1
    assert retriever.calls.count("rewritten bridge query") == 1
    assert result["answer"] == "A connects through the bridge. [1][2]"
    assert len(result["citations"]) == 2
    assert result["generated_claims"] == result["grounded_claims"] == 1
    assert result["citation_references"] == result["valid_citation_references"] == 2
    assert [group["title"] for group in result["context_groups"]] == ["a", "b"]


@pytest.mark.asyncio
async def test_weak_direct_retrieval_escalates_to_complex_route():
    class DirectModels(AgenticModels):
        async def complete(self, messages, **kwargs):
            if "Classify and expand" in messages[0]["content"]:
                return '{"route":"direct","expanded_query":"connection","terms":[]}'
            return await super().complete(messages, **kwargs)

    class Retriever:
        def __init__(self):
            self.routes = []

        async def retrieve(self, query, route, auth, limit):
            self.routes.append(route)
            score = 0.1 if route == Route.DIRECT else 0.9
            return [_evidence("a", "supported A evidence", score)]

    retriever = Retriever()
    result = await RagPipeline(retriever, Settings(), DirectModels()).run(
        "What is the connection?", AuthContext(subject="user", tenant_id="tenant")
    )

    assert retriever.routes[0] == Route.DIRECT
    assert Route.MULTIHOP in retriever.routes
    assert result["escalated"] is True


@pytest.mark.asyncio
async def test_llm_override_reclassifies_a_query_the_keyword_router_defaults_to_direct():
    """The keyword router's fallback for anything it doesn't recognize is DIRECT --
    that's exactly the case that most needs a second look, not the one case the old
    gate (`route != Route.DIRECT`) skipped it for."""

    class Models(AgenticModels):
        async def complete(self, messages, **kwargs):
            if "Classify and expand" in messages[0]["content"]:
                return (
                    '{"route":"comparison","expanded_query":"compare A and B",'
                    '"terms":["a","b"]}'
                )
            return await super().complete(messages, **kwargs)

    class Retriever:
        routes: list[Route] = []

        async def retrieve(self, query, route, auth, limit):
            self.routes.append(route)
            return [_evidence("a", "supported A evidence")]

    retriever = Retriever()
    keyword_route = classify("Does product A outsell product B this quarter?")
    assert keyword_route == Route.DIRECT

    await RagPipeline(retriever, Settings(), Models()).run(
        "Does product A outsell product B this quarter?",
        AuthContext(subject="user", tenant_id="tenant"),
    )

    assert Route.COMPARISON in retriever.routes


@pytest.mark.asyncio
async def test_classify_fallback_to_keyword_route_logs_reason(caplog):
    class BrokenModels:
        async def complete(self, messages, **kwargs):
            return "not json"

    retriever = LeafRetriever()
    with caplog.at_level("WARNING"):
        await RagPipeline(retriever, Settings(), BrokenModels()).run(
            "compare A and B", AuthContext(subject="user", tenant_id="tenant")
        )

    fallbacks = {
        record.node: record.reason
        for record in caplog.records
        if record.getMessage() == "fallback_used"
    }
    assert fallbacks["classify"] == "ValidationError"


@pytest.mark.asyncio
async def test_critique_falls_back_instead_of_crashing_on_token_limit_truncation(caplog):
    """A truncated completion (finish_reason='length') raises openai.LengthFinishReasonError,
    not ValueError/json.JSONDecodeError -- the whole run must still complete via the
    lexical-fallback grader, not propagate and kill the durable run."""

    class TruncatedCritiqueModels(AgenticModels):
        async def complete(self, messages, **kwargs):
            if "Judge each candidate" in messages[0]["content"]:
                import openai

                fake_completion = type("FakeCompletion", (), {"usage": None})()
                raise openai.LengthFinishReasonError(completion=fake_completion)
            return await super().complete(messages, **kwargs)

    retriever = LeafRetriever()
    with caplog.at_level("WARNING"):
        result = await RagPipeline(
            retriever, Settings(max_leaf_retries=2), TruncatedCritiqueModels()
        ).run("Find the multi-hop connection", AuthContext(subject="user", tenant_id="tenant"))

    assert result["status"] == "complete"
    fallbacks = {
        record.node: record.reason
        for record in caplog.records
        if record.getMessage() == "fallback_used"
    }
    assert fallbacks["critique"] == "LengthFinishReasonError"


@pytest.mark.asyncio
async def test_synthesis_router_expansion_is_used_for_retrieval():
    class Models:
        async def complete(self, messages, **kwargs):
            if "Classify and expand" in messages[0]["content"]:
                return (
                    '{"route":"synthesis","expanded_query":"RAPTOR deployment topology",'
                    '"terms":["RAPTOR","topology"]}'
                )
            return (
                '{"claims":[{"text":"RAPTOR is deployed.","evidence_ids":["a"]}],"unsupported":[]}'
            )

    class Retriever:
        query = None

        async def retrieve(self, query, route, auth, limit):
            self.query = query
            return [_evidence("a", "RAPTOR deployment topology is active.")]

    retriever = Retriever()
    await RagPipeline(retriever, Settings(), Models()).run(
        "Summarize deployment", AuthContext(subject="user", tenant_id="tenant")
    )

    assert retriever.query == "RAPTOR deployment topology"


@pytest.mark.asyncio
async def test_direct_lookup_skips_expansion_plan_and_critique_model_calls():
    """DIRECT still skips its own expand/plan/critique LLM calls (the fast-path
    invariant) -- only classify (now always-on) and generate call the model."""

    class Models:
        calls = []

        async def complete(self, messages, **kwargs):
            self.calls.append(messages[0]["content"])
            if "Classify and expand" in messages[0]["content"]:
                return '{"route":"direct","expanded_query":"clustering method","terms":[]}'
            return '{"claims":[{"text":"GMM is used.","evidence_ids":["a"]}],"unsupported":[]}'

    class Retriever:
        async def retrieve(self, query, route, auth, limit):
            assert query == "clustering method"
            return [_evidence("a", "GMM is used.")]

    models = Models()
    await RagPipeline(Retriever(), Settings(), models).run(
        "Which clustering method is used?", AuthContext(subject="user", tenant_id="tenant")
    )

    assert len(models.calls) == 2
    assert "Classify and expand" in models.calls[0]
    assert "atomic factual claims" in models.calls[1]


@pytest.mark.asyncio
async def test_critic_bounds_each_leaf_before_sending_text_to_model():
    class Models:
        user_prompt = ""

        async def complete(self, messages, **kwargs):
            self.user_prompt = messages[-1]["content"]
            return '{"decisions":[]}'

    models = Models()
    pipeline = RagPipeline(object(), Settings(max_evidence_per_leaf=2), models)
    result = await pipeline._critique(
        {
            "route": Route.MULTIHOP,
            "tenant_id": "tenant",
            "groups": [],
            "model_calls": 0,
            "leaf_states": [
                {
                    "id": "leaf",
                    "question": "Find the link",
                    "status": "retrieved",
                    "candidates": [
                        _evidence("a", "first").model_dump(mode="json"),
                        _evidence("b", "second").model_dump(mode="json"),
                        _evidence("c", "must not reach critic").model_dump(mode="json"),
                    ],
                }
            ],
        }
    )

    assert result["critic_candidates"] == 2
    assert "Candidate ID: c" not in models.user_prompt


@pytest.mark.asyncio
async def test_leaf_retrieval_runs_independently_in_parallel():
    class Retriever:
        started = 0
        both_started = asyncio.Event()

        async def retrieve(self, query, route, auth, limit):
            self.started += 1
            if self.started == 2:
                self.both_started.set()
            await asyncio.wait_for(self.both_started.wait(), 0.2)
            return [_evidence(query, query)]

    pipeline = RagPipeline(Retriever(), Settings())
    result = await pipeline._retrieve(
        {
            "route": Route.MULTIHOP,
            "tenant_id": "tenant",
            "user_id": "user",
            "groups": [],
            "retrieval_calls": 0,
            "attempts": {},
            "leaf_states": [
                {"id": "a", "question": "a", "query": "a", "attempts": 0, "status": "pending"},
                {"id": "b", "question": "b", "query": "b", "attempts": 0, "status": "pending"},
            ],
        }
    )

    assert result["retrieval_calls"] == 2


@pytest.mark.asyncio
async def test_direct_lookup_keeps_only_two_best_contexts():
    pipeline = RagPipeline(object(), Settings(max_evidence_per_leaf=8))
    result = await pipeline._critique(
        {
            "route": Route.DIRECT,
            "tenant_id": "tenant",
            "groups": [],
            "leaf_states": [
                {
                    "id": "leaf",
                    "question": "exact lookup",
                    "status": "retrieved",
                    "candidates": [
                        _evidence("a", "exact lookup first", 0.9).model_dump(mode="json"),
                        _evidence("b", "exact lookup second", 0.8).model_dump(mode="json"),
                        _evidence("c", "exact lookup third", 0.7).model_dump(mode="json"),
                    ],
                }
            ],
        }
    )

    assert [item["id"] for item in result["accepted_evidence"]] == ["a", "b"]


@pytest.mark.asyncio
async def test_generation_deduplicates_repeated_parent_context():
    class Models:
        prompt = ""

        async def complete(self, messages, **kwargs):
            self.prompt = messages[-1]["content"]
            return '{"claims":[{"text":"Supported.","evidence_ids":["a"]}],"unsupported":[]}'

    models = Models()
    first = _evidence("a", "leaf a").model_copy(update={"context_text": "shared parent"})
    second = _evidence("b", "leaf b").model_copy(update={"context_text": "shared parent"})
    await RagPipeline(object(), Settings(), models)._generate(
        {
            "route": Route.SYNTHESIS,
            "query": "Summarize",
            "tenant_id": "tenant",
            "accepted_evidence": [
                first.model_dump(mode="json"),
                second.model_dump(mode="json"),
            ],
            "context_groups": [{"title": "doc", "ids": ["a", "b"]}],
        }
    )

    assert models.prompt.count("shared parent") == 1


@pytest.mark.asyncio
async def test_generation_refuses_when_model_declines_to_synthesize_a_connection():
    """Real evidence for each half of a compound question can still leave the
    connection itself unstated -- an empty claims list (with reasons in
    `unsupported`) must read as a refusal, not as an empty non-answer. Without
    this, `all(... for claim in [])` is vacuously True and the pipeline would
    treat zero claims as a successful, grounded, zero-content answer."""

    class Models:
        async def complete(self, messages, **kwargs):
            return (
                '{"claims":[],'
                '"unsupported":["no evidence states the connection between A and B"]}'
            )

    result = await RagPipeline(object(), Settings(), Models())._generate(
        {
            "route": Route.MULTIHOP,
            "query": "How is A connected to B?",
            "tenant_id": "tenant",
            "accepted_evidence": [_evidence("a", "A is a company.").model_dump(mode="json")],
            "context_groups": [{"title": "doc", "ids": ["a"]}],
        }
    )

    assert "could not find enough authorized evidence" in result["answer"]
    assert "connection between A and B" in result["answer"]
    assert result["grounded_claims"] == 0
    assert result["citation_verification"] == "unsupported"


@pytest.mark.asyncio
async def test_generate_refuses_instead_of_dumping_evidence_when_grounded_answer_unparseable():
    """Root cause of the null_query refusal-accuracy bug: MiniMax-M3's
    <think>-wrapped output (fixed separately in llm.py) failing to parse used
    to fall through to _numbered_fallback, which dumps whatever evidence was
    retrieved -- even if it's about a completely different topic -- as if it
    were a real answer. A parse failure must be conservative (refuse), not
    permissive (show unrelated evidence)."""

    class Models:
        async def complete(self, messages, **kwargs):
            return "<think>truncated reasoning with no closing tag"

    result = await RagPipeline(object(), Settings(), Models())._generate(
        {
            "route": Route.MULTIHOP,
            "query": "Unanswerable question",
            "tenant_id": "tenant",
            "accepted_evidence": [
                _evidence("a", "unrelated evidence text").model_dump(mode="json")
            ],
            "context_groups": [{"title": "doc", "ids": ["a"]}],
        }
    )

    assert "could not find enough authorized evidence" in result["answer"]
    assert "unrelated evidence text" not in result["answer"]
    assert result["answer_source"] == "unverifiable"


@pytest.mark.asyncio
async def test_generate_refuses_when_citations_reference_evidence_outside_accepted_set():
    """A model can return well-formed JSON that still cites an evidence_id we
    never accepted (hallucinated or stale reference) -- this must also refuse
    rather than fall through to a raw evidence dump."""

    class Models:
        async def complete(self, messages, **kwargs):
            return (
                '{"claims":[{"text":"Some claim.","evidence_ids":["not-accepted"]}],'
                '"unsupported":[]}'
            )

    result = await RagPipeline(object(), Settings(), Models())._generate(
        {
            "route": Route.MULTIHOP,
            "query": "Some question",
            "tenant_id": "tenant",
            "accepted_evidence": [_evidence("a", "real evidence text").model_dump(mode="json")],
            "context_groups": [{"title": "doc", "ids": ["a"]}],
        }
    )

    assert "could not find enough authorized evidence" in result["answer"]
    assert "real evidence text" not in result["answer"]
    assert result["answer_source"] == "unverifiable"


@pytest.mark.asyncio
async def test_generate_still_uses_numbered_fallback_when_no_model_is_configured():
    """_numbered_fallback (raw evidence dump) stays correct for the genuinely
    different case of no LLM configured at all -- there's no model to ask for
    a grounded answer, so showing the retrieved evidence directly is the best
    available behavior. This must not regress from the _generate changes."""

    result = await RagPipeline(object(), Settings(), None)._generate(
        {
            "route": Route.MULTIHOP,
            "query": "Some question",
            "tenant_id": "tenant",
            "accepted_evidence": [_evidence("a", "real evidence text").model_dump(mode="json")],
            "context_groups": [{"title": "doc", "ids": ["a"]}],
        }
    )

    assert "real evidence text" in result["answer"]
    assert result["answer_source"] == "fallback"


@pytest.mark.asyncio
async def test_generate_no_evidence_early_return_reports_its_answer_source():
    result = await RagPipeline(object(), Settings(), None)._generate(
        {
            "route": Route.MULTIHOP,
            "query": "Some question",
            "tenant_id": "tenant",
            "accepted_evidence": [],
            "context_groups": [],
        }
    )

    assert result["answer_source"] == "no_evidence"


@pytest.mark.asyncio
async def test_generate_propagates_a_transport_error_from_the_grounded_answer_call():
    """_grounded_answer's try/except only guards the JSON parse, not the
    completion call itself -- a real transport error must still kill the
    durable run rather than silently degrading into a refusal."""

    class Models:
        async def complete(self, messages, **kwargs):
            raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        await RagPipeline(object(), Settings(), Models())._generate(
            {
                "route": Route.MULTIHOP,
                "query": "Some question",
                "tenant_id": "tenant",
                "accepted_evidence": [_evidence("a", "evidence").model_dump(mode="json")],
                "context_groups": [{"title": "doc", "ids": ["a"]}],
            }
        )


@pytest.mark.asyncio
async def test_answer_source_survives_the_full_graph_run_not_just_generate_directly():
    """AgentState (schemas/domain.py) is the TypedDict LangGraph uses to define
    its state channels -- a node returning a key absent from that schema gets
    silently dropped when merged into the graph's final state. Calling
    _generate() directly (as the other tests here do) can't catch that gap;
    only a full RagPipeline.run() through the actual graph can."""

    class Models(AgenticModels):
        async def complete(self, messages, **kwargs):
            if "atomic factual claims" in messages[0]["content"]:
                return "not json"
            return await super().complete(messages, **kwargs)

    result = await RagPipeline(LeafRetriever(), Settings(max_leaf_retries=2), Models()).run(
        "Find the multi-hop connection", AuthContext(subject="user", tenant_id="tenant")
    )

    assert result["answer_source"] == "unverifiable"
