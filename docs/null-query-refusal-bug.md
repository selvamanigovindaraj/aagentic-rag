# Bug: `null_query` cases wrongly answered instead of refused

**Status:** Investigated, root cause confirmed by direct reproduction. No code changes made
(per request) — findings and a suggested direction are documented here for a follow-up fix.

## Symptom

Generation eval (250 MultiHop-RAG-derived queries, a historical dev-only reproduction environment with Weaviate run locally instead of Weaviate Cloud plus Neo4j/Postgres, `evaluation/generation_eval_local250_results.jsonl`) failed its threshold overall (`refusal_accuracy: 0.852` vs. `0.90` required). The failure concentrates almost entirely in the `null_query` bucket:

| question_type | count | refusal_accuracy |
|---|---|---|
| inference_query | 80 | 0.988 |
| comparison_query | 80 | 0.950 |
| temporal_query | 60 | 0.850 |
| **null_query** | 30 | **0.233** |

`null_query` is MultiHop-RAG's label for questions that are genuinely unanswerable from the
corpus (see `evaluation/build_multihop_dataset.py`'s module docstring) — every case has
`expect_refusal: true`. 23 of 30 (77%) got a real-looking answer instead of the expected refusal.

Notably, `grounded_claim_rate: 1.0` and `citation_validity: 1.0` for the whole run gave no hint
of this — see "Why the eval's own metrics didn't catch it" below.

## Reproduction

Ran one failing case directly against the live pipeline (`RagPipeline.run`), no eval harness involved:

```text
Query: "Considering the information from a New York Times article and a Wall Street Journal
report on Ryan McInerney, what is the first letter of the city where the company he leads is
headquartered, which also announced a significant financial technology investment as per the
New York Times, and is facing regulatory scrutiny as mentioned in the Wall Street Journal?"
```

Result: `route: "multi_hop"`, decomposed into 5 subquestions. The accepted evidence was an
article titled *"The inside story of Dave Clark's tumultuous last days at Flexport"* (score
`0.4375`) — about Flexport CEO Ryan Petersen and ex-COO Dave Clark. It has nothing to do with
"Ryan McInerney." The final `answer` was a verbatim dump of that unrelated article's text with
`[1][2]` citation markers — not a refusal, not even a coherent attempt to answer the question
asked.

Two `fallback_used` warnings were logged during this single run (visible in stdout because
`configure_logging()`'s `ExtraFieldsFormatter` isn't wired into this eval script, so the log
line's `extra={"node": ..., "reason": ...}` fields — which would normally identify exactly
which node/reason — are silently dropped by Python's default handler; see the CLAUDE.md
logging invariant. This makes per-case root-causing harder in this script than it should be,
but manual reproduction bypassed that).

## Root cause: two compounding gaps

**1. The evidence-acceptance gate has no relevance floor.**

`backend/app/components/document_grader.py::grade()` delegates entirely to
`backend/app/components/reranker.py::rerank()`:

```python
def rerank(query, evidence, limit):
    terms = {t for t in re.findall(r"[a-z0-9-]+", query.lower()) if len(t) > 2}
    relevant = [e for e in evidence if terms & set(re.findall(r"[a-z0-9-]+", e.text.lower()))]
    return sorted(relevant, key=lambda item: item.score, reverse=True)[:limit]
```

This keeps any evidence chunk sharing *any* 3+ character token with the query — which includes
ordinary stopwords ("the", "and", "was", "company") — then just sorts by score and truncates to
`limit`. There is no minimum-score/minimum-relevance threshold. When the true answer doesn't
exist in the corpus (by construction, for `null_query`), this still returns the *least-bad*
available candidate as if it were valid evidence, however unrelated.

**2. `_generate`'s fallback path treats "JSON didn't parse" as "show the evidence," not "be conservative."**

`backend/app/services/rag_pipeline.py`:

- `_generate` (line 436) calls `_grounded_answer`, which prompts the model for a structured
  `GroundedAnswer` (`claims` + `evidence_ids`, plus an `unsupported` list —
  `backend/app/schemas/agent.py:55-57`). This schema is *specifically designed* to let a
  correctly-behaving model decline: empty `claims` + populated `unsupported` routes to
  `_declined_answer` (line 483-489), which is the correct refusal.
- If the raw LLM output fails `GroundedAnswer.model_validate_json(raw)` (line 480-484,
  catching `ValueError | json.JSONDecodeError | OpenAIError`), `_grounded_answer` returns
  `None` — this is the sanctioned "malformed structured response falls back inside its graph
  node" pattern from CLAUDE.md.
- But in `_generate`, when `grounded is None`, **none** of the `if grounded and ...` branches
  match (line 450-458), and execution falls through unconditionally to
  `return self._numbered_fallback(state, evidence)` (line 459).
- `_numbered_fallback` (line 521-524) concatenates the raw text of *every currently-accepted*
  evidence chunk as the final answer, with numbered citations — with no relevance check, no
  attempt at refusal, and no awareness that the evidence might be entirely off-topic.

So the one call that was actually designed to catch this exact situation (irrelevant evidence →
model says "unsupported" → refuse) is exactly the call whose failure-to-parse throws away that
safety net and substitutes a much more permissive one. A JSON-parse hiccup silently converts a
would-be-correct refusal into a confident-looking wrong answer.

## Why the eval's own metrics didn't catch it

`_numbered_fallback` calls `_generation_result` with `claims=len(evidence)` and then sets
`grounded_claims=claims` and `valid_citation_references=references` **by construction** (line
524, 543-546) — not by verifying anything against evidence content. The eval script's `grounded`
and `citations_valid` fields (`evaluation/generation_eval.py::_case_result`) compare these two
numbers to each other, so a fallback answer is *tautologically* "100% grounded" and "100% valid
citations" regardless of whether the evidence is remotely relevant. Only `refusal_accuracy`
(whether the pipeline refused when it should have) exposes this bug — which is exactly why the
run's `grounded_claim_rate: 1.0` / `citation_validity: 1.0` looked clean while `null_query`
refusal accuracy was 0.233.

## Scope note

`fallback_used` was logged 332 times across the full 250-query run (any node — classify, expand,
plan, critique, generate — can independently trigger it), so this isn't unique to `null_query`;
it likely also contributes to `comparison_query`'s low answer_accuracy (0.316) in the same run,
though that wasn't separately reproduced here. The `null_query` bucket is simply where the
consequence is most visible, because there the correct behavior (refuse) and the fallback's
actual behavior (dump evidence) are furthest apart.

## Suggested direction (not implemented — no code changes made per request)

Two independent, complementary fixes would address this without touching the sanctioned
"fallback inside the node" pattern itself:

1. Give `reranker.rerank()` (or `document_grader.grade()`) an actual minimum-relevance floor,
   not just lexical presence + sort-by-score-then-truncate, so a corpus with no real answer
   yields *no* accepted evidence rather than the least-bad candidate.
2. Make `_generate`'s fallback conservative in the specific case where `_grounded_answer`
   returned `None` due to a parse failure (as opposed to `self.models` being absent, which is
   `_numbered_fallback`'s other caller) — e.g. retry the grounded-claims call once, or fall back
   to refusal rather than a raw evidence dump, so a JSON-parsing hiccup can't silently overturn
   what should have been a refusal.
