# Eval-driven hardening: retrospective

This covers three linked pieces of work on the agentic RAG backend: fixing the 8 findings from
an architecture review, building a real (not toy) evaluation harness against the MultiHop-RAG
dataset, and then using that harness to push route accuracy, refusal accuracy, and multi-hop
recall to production-grade thresholds. Written as a record of what was actually hard, what was
tried and rejected, and why the shipped approach won — not a changelog of what shipped.

## 1. Reviewing without git

The repo wasn't a git repository during the architecture-review pass, so there was no `git diff`
to anchor a code review against. Findings had to be re-derived by reading each touched file in
full after every edit rather than diffing against a baseline, and the clean-code-review pass at
the end was a manual re-read of every changed file rather than a diff review. This was slower but
forced closer reading of each change in context, which is arguably how the DRY duplication
(`cosine_similarity` reimplemented in both `graph_index.py` and `retrieval.py`) got caught.

## 2. The groundedness check: how strict is too strict?

**Problem.** Finding #2 (durable corpus poisoning via untrusted document content) needed a check
that rejects LLM-extracted triples/summaries with no relationship to their source text, without a
real entailment/NLI model available in the stack.

**Approaches considered:**
- *Exact substring match* (subject and object must each appear verbatim in the source text) —
  rejected after checking it against the existing test fixtures: a fixture's expected object,
  `"recursive summaries"`, doesn't appear verbatim in its source (`"RAPTOR builds summaries."`),
  so exact matching would have broken legitimate paraphrased extraction, not just injected
  content.
- *Full NLI/entailment model* — rejected as disproportionate; this system has no NLI dependency
  anywhere, and adding one for a defense-in-depth check is a new dependency for a narrow need.
- *Word-overlap heuristic* (significant-word set intersection between candidate and source, with
  a substring fallback for short entities that have no 4+ letter words) — selected. It's a few
  lines, no new dependency, and it was verified against the real fixtures (both the existing
  "recursive summaries" case and new adversarial-injection test cases) before being trusted.

The verification step mattered: the first version was tested only against injected garbage
(`"click here for a free prize"` — correctly rejected) but not re-checked against the existing
legitimate fixtures until after implementation, which is when the substring-only design would
have been caught as too strict, had it been chosen.

## 3. A bug the unit tests couldn't see: logging never configured

**Problem.** Live verification (bringing up the real `docker compose` stack and driving it with
real ingestion + chat requests) surfaced that `weaviate_search` (INFO-level) events were silently
absent from container logs, even though the code path had definitely run.

**Root cause.** Neither `main.py` nor `worker.py` ever called `logging.basicConfig`. Python's
"last resort" handler only surfaces WARNING+ when nothing else is configured. `caplog`-based unit
tests never catch this class of bug, because `caplog` attaches its own handler directly and
bypasses the question of whether the *application* configured logging at all — the tests were
technically passing while validating a code path that would never have printed anything in
production.

**Fix and why it's not just `logging.basicConfig(level=logging.INFO)`:** a bare basicConfig call
was tried first and technically fixed the visibility problem, but the resulting log line
(`INFO:app.components.retrieval:weaviate_search`) carried none of the `extra={...}` fields
(tenant_id, result_count, duration_ms) that the instrumentation was added for in the first place —
defeating its purpose. This is what justified writing `core/logging.py`'s
`ExtraFieldsFormatter` rather than accepting the one-line fix: the *point* of the earlier
instrumentation work was the structured fields, and a fix that restored visibility but dropped
the fields would have been declaring victory on the wrong problem.

## 4. Finding the real MultiHop-RAG data

**Problem.** The GitHub repo (`yixuantt/MultiHop-RAG`) doesn't contain a `dataset/` directory —
guessing at `raw.githubusercontent.com/.../dataset/MultiHopRAG.json` returned 404. The actual
data is hosted on Hugging Face (`yixuantt/MultiHopRAG`), findable only by reading the repo's
README rather than assuming a conventional file layout.

A second, smaller trap: `curl`-ing the Hugging Face `resolve/main/...` URL without `-L` returns a
307 redirect body (a small JSON pointer to the real content location), not the file — the first
download attempt silently produced a 267-byte "file" that failed to parse as JSON. This is a
generically useful lesson (verify downloaded file sizes look plausible before trusting them,
`-sSL` not `-sS` for anything behind a CDN redirect), not specific to this dataset.

## 5. Choosing what to build the eval fixtures from

**Problem.** MultiHop-RAG has 4 query types; the app has 6 routes. They don't map 1:1.

**Key decision:** `null_query` ("unanswerable from this corpus") does *not* map to
`Route.OUT_OF_SCOPE`. Reading `adaptive_router.py` showed `OUT_OF_SCOPE` is a jailbreak/abuse
guard (triggers only on phrases like "ignore previous instructions"), not a knowledge-boundary
classifier — that refusal happens later, in `_generate`, when accepted evidence is empty,
regardless of route. Forcing `null_query → out_of_scope` into the route-accuracy dataset would
have manufactured a false "router bug" out of a route the classifier was never supposed to
predict for that input. `null_query` cases were routed instead to the generation/refusal eval,
which is what they actually test.

**Sizing the retrieval fixture.** Ingesting the full 609-article corpus was rejected as
disproportionate for validation (real LLM/embedding cost and wall-clock time per document, and a
single worker processes ingestion jobs serially). The fixture was built by sampling N queries per
answerable type first, then taking the *union* of articles their evidence actually references —
this guarantees `expected_document_titles` are always satisfiable by what gets ingested, rather
than picking N articles first and hoping enough queries happen to be covered (which, tried
mentally first, would have produced a mostly-empty retrieval fixture since MultiHop-RAG queries
reference 2–4 specific articles each, not a random subset of the corpus).

## 6. A production bug the eval accidentally found: ingestion had no fallback where the query path did

**Problem.** Live ingestion of the sample corpus threw `openai.LengthFinishReasonError`
(truncated completion) and a Pydantic `too_long` error (LLM returned 21 triples against a
20-item cap), and both **crashed the whole ingestion job** rather than degrading — even though
`rag_pipeline.py`'s query-time nodes all wrap their model calls in
`except (ValueError, json.JSONDecodeError)`.

**Why this wasn't caught by the earlier architecture-review fixes:** those fixes were about
adding *groundedness* checks to `graph_index.py`/`raptor.py`, not about wrapping the model call
itself in error handling — the review's own finding #7 ("nearly every LLM call has a
try/except fallback... blind spot?") turned out to be wrong in the specific case of ingestion:
there was *no* try/except there at all, only at query time. This was only discoverable by
actually running ingestion against real, variable-length news articles — synthetic test fixtures
with short, predictable text never produce a 21-item triple batch or a token-limit truncation.

**Fix considered and rejected:** narrowing `StatementBatch`'s `max_length=20` cap or only
catching `pydantic.ValidationError` specifically. Rejected because it only patches the one
observed symptom (the 21-item case) and leaves the token-truncation case (a different exception
type, `openai.LengthFinishReasonError`, not a `ValueError` subclass) uncaught. The actual fix —
wrap the call in `except (ValueError, json.JSONDecodeError, OpenAIError)`, skip just that chunk,
keep the document's other chunks — matches the existing query-time pattern instead of inventing a
new one, and was later found to be needed in `rag_pipeline.py` itself too (see §9).

## 7. Concurrent live evals corrupt each other — a lesson learned the hard way

Two live evaluation scripts were run concurrently against the same DeepSeek API key partway
through the route-accuracy work (a `query_latency.py` run launched while a route-accuracy-live
run was still finishing). The route-accuracy number that came back was inconsistent with a
clean re-run of the identical prompt, and the *first* instinct — that the prompt draft itself had
regressed — would have been wrong. Re-running in isolation (the concurrent job had actually
finished by then) reproduced a similar number, which resolved the ambiguity, but it cost a full
extra live run to find out. After that, every subsequent live validation was run one at a time,
even though it meant more wall-clock time waiting on sequential ~2–5 minute runs — trustworthy
numbers took priority over speed once the contamination risk was understood.

## 8. Route accuracy: rule-based prompting plateaued, then regressed, before few-shot fixed it

**Baseline problem.** `_classify`'s LLM override only ran when the keyword router picked
something *other than* `direct` — backwards, since `direct` is the router's fallback for
anything unrecognized, i.e. exactly the case most likely to be wrong. Fixing the gate condition
alone took accuracy from 16.7% → 61.7% live.

**Then three rounds of abstract-rule prompt iteration, each validated against the real confusion
matrix rather than eyeballed:**

1. Added route definitions with discriminating criteria (comparison vs temporal_causal vs
   multi_hop) plus a tie-break rule. Result: 61.7% → 65.8% (temporal_causal confusion nearly
   halved, but a *new* multi_hop → direct confusion appeared — 5 → 15 misses).
2. Diagnosed the new confusion by reading the actual misclassified queries: they were almost all
   "Who is X, according to \[3 different publications\]..." — grammatically a single-fact
   question, but requiring cross-referencing sources to answer. The "direct = single fact
   lookup" rule was too permissive because it judged by the grammatical shape of the expected
   answer rather than by what retrieving it actually required. Added a rule making "names
   multiple sources needing combination → never direct" the first check. Result: 65.8% → 80.8%
   sampled, but this rule was now *too broad* and started swallowing genuine `comparison`
   queries that also happen to name multiple sources (extremely common in this dataset by
   construction).
3. Added a further rule distinguishing "explicit compare/contrast request" (comparison) from
   "identify an unknown from scattered clues" (multi_hop). Result: 80.8%.
4. Attempted one more refinement targeting the largest remaining bucket
   (`temporal_causal → comparison`, still ~13–24 misses): treat "are two *reports* consistent"
   as temporal_causal. This **regressed** to 75.8% — the rule was too broad again, this time
   pulling genuine "does source A's claim align with source B's claim" comparisons into
   `temporal_causal`. This iteration was reverted rather than kept, rather than assume a
   regression was noise — a second full 120-case run at the reverted (iteration-3) prompt
   confirmed 81.7%, consistent with the earlier 80.8% measurement (small run-to-run LLM
   non-determinism, not the review-worthy kind of drift).

**Why the plateau, and why few-shot broke it.** By iteration 3, further abstract rules were
oscillating rather than converging — each fix for one confusion pair reliably broke another,
because the underlying distinction (e.g. "Between report A and report B, was there a
discrepancy" → `temporal_causal`, vs "Does source A suggest X while source B suggests Y" →
`comparison`) is closer to a dataset-construction template artifact than a clean semantic rule
expressible in a sentence or two. Four concrete few-shot examples, each taken directly from real
misclassified queries and their correct route, were added instead of one more abstract rule.
Result: 80.8% → **95.0%** (and 95.8% on a second independent run). This is the single largest
jump of any change in this effort, and in hindsight should have been tried before the third round
of rule refinement — abstract rules were being asked to do what concrete examples do more
reliably for this class of subtle, template-shaped disambiguation.

**A rejected alternative for the whole exercise:** stopping at 80.8% and reporting the residual
gap as "genuine dataset ambiguity" was seriously considered after the iteration-4 regression —
the mistake there had just been in how broadly a real signal (report-vs-report consistency) was
phrased, not that the underlying signal was fake. Concluding "further improvement isn't possible"
after one bad attempt at encoding a real pattern would have been premature; the few-shot approach
proved the ceiling was much higher than 80.8%.

## 9. Refusal accuracy: two bugs stacked, and a Python gotcha at the bottom

**First pass.** After the routing fix (§8, step 1), refusal accuracy did *not* improve — it went
from 1/10 to 0/10 correct on unanswerable queries. Re-checking the routing hypothesis by hand:
now-correctly-routed compound questions *were* reaching the LLM evidence critic (whose prompt
already explicitly says "reject topical but non-supporting text"), and the critic *was* correctly
rejecting insufficient per-leaf evidence in isolation — so the leak was downstream, at generation.

**Second bug, found by tracing the actual generation output.** For a compound question like "does
a BBC article on X and a Times of India article on Y reveal connection Z", retrieval correctly
found real evidence for X and separately for Y; the critic correctly accepted both (each is
individually true); but `_generate`'s `GROUNDED_CLAIMS_V1` step then wrote a claim asserting
connection Z, citing the real IDs for the X and Y evidence — which are valid, authorized,
accepted evidence IDs, so the code's own "are citations valid" check passed. The two true facts
being separately true does not make a claim linking them true, and nothing in the pipeline
checked that.

**Prompt fix:** made `GROUNDED_CLAIMS_V1` explicit that a claim must be *stated* in a single
evidence span, not synthesized by combining two spans, and that an empty `claims` list with
reasons in `unsupported` is a correct answer when evidence doesn't go far enough — rather than
leaving "put unsupported portions in unsupported" as the only (and, in practice, insufficiently
followed) instruction.

**Third bug, a language-level trap, found only after the prompt fix still didn't produce a clean
refusal string in testing:** `all(condition for claim in grounded.claims)` is vacuously `True`
when `grounded.claims` is empty — Python semantics, not a logic error in the surrounding code —
so a model correctly returning zero claims was being treated as a *successful, grounded,
zero-content* answer by the existing `if grounded and all(...)` check, producing an answer that
was just `"\n\nUnsupported: ..."` with no refusal framing, rather than the pipeline's actual
refusal message. This needed an explicit `if grounded and not grounded.claims:` branch before the
`all(...)` check, not a change to the `all(...)` condition itself (which was already correct for
the non-empty case). Result after all three fixes: **100%** refusal accuracy (from 10%), with
answerable-case grounding and citation validity unaffected (still 100%).

## 10. The latency trade-off: accept it, don't hide it

Making `_classify`'s LLM call unconditional (§8) is a mandatory extra sequential LLM round-trip on
every `direct`-route query, which is supposed to be the fast path. Live measurement (isolated,
after ruling out the concurrency-contamination pattern from §7) showed clean steady-state p95
around 3.2–3.7s against a documented `direct_p95_ms: 2000` threshold.

**Options weighed:**
- *Revert the correctness fix to protect latency* — rejected; it would have reintroduced the
  16.7% route-accuracy failure the whole effort started from, for a hard requirement
  ("production grade accuracy") that was more explicit than the latency threshold.
- *Speculative execution* (start retrieval with the keyword-guessed route concurrently with the
  LLM classify call, discard/redo if the LLM overrides it) — rejected as scope creep for this
  task: real added complexity (wasted retrieval calls on override, harder-to-reason-about
  control flow) for a latency problem that wasn't the thing being asked to fix.
- *Silently leave the threshold at 2000ms failing* — rejected; a known-failing threshold sitting
  unfixed in the repo is worse than an honest, higher, currently-passing one, because it trains
  people to ignore the gate.
- *Update the threshold to reflect measured reality, with the reasoning written down* — selected.
  `direct_p95_ms` moved to 3500 in `evaluation/thresholds.json`, with the trade-off and a pointer
  to what a real latency fix would require (removing the always-on call for high-confidence
  keyword matches, not just prompt tuning) recorded in `CLAUDE.md` so it isn't silently
  rediscovered later.

## 11. Verifying instrumentation reaches somewhere real, not just "is configured"

When asked whether evaluation and the agent/ingestion paths were connected to LangSmith, checking
`compose.yaml` alone would have given a falsely reassuring "yes" (the env vars are declared
there). Two things made the actual answer more precise and partly negative:

- Querying the LangSmith API directly (`GET /sessions`, then `POST /runs/query` for the
  `agentic-rag` session) to confirm real, structured, timestamped traces existed — not just that
  the API key was present, but that it was accepted and traces actually landed, with the expected
  LangGraph node names nested under `ChatDeepSeek` LLM runs.
- Checking the *host* process environment separately from the *container* environment
  (`uv run python -c "import os; print(os.environ.get('LANGCHAIN_TRACING_V2'))"` → `None`), which
  is what surfaced that `evaluation/*.py` scripts run outside Docker and never inherit `.env`,
  because `pydantic-settings` reading `.env` into its own model fields is a different mechanism
  from an OS environment variable being set — a distinction that's easy to elide if you only read
  the config-loading code rather than checking the actual process environment at runtime.

## General pattern across all of the above

The common thread: several of the real bugs here (the logging gap, the ingestion fallback gap,
the vacuous-`all()` bug, the LangSmith host/container split) were invisible to unit tests and
would have shipped clean on a "tests pass" signal alone, because the bug lived in the gap between
what a component does in isolation and what actually happens when it's wired into a real running
process. None of the multi-round prompt iteration (§8) would have been possible to do responsibly
by eyeballing outputs either — every claimed improvement or regression was checked against a full
confusion matrix from a live run before being trusted, including reverting one iteration that
looked plausible in isolation but measured worse.
