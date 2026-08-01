# Pitfalls and lessons

A running log of concrete pitfalls hit while developing this project, and the generalizable
lesson each one left behind. Unlike `eval-hardening-retrospective.md` (a narrative writeup of one
body of work) or `null-query-refusal-bug.md` (a deep-dive on one bug), this doc is meant to grow
incrementally — add a short entry whenever a bug, a wrong assumption, or a debugging dead-end
teaches something that will bite again in a different disguise.

**When to add an entry:** after fixing a bug (or hardening config/infra) where the root cause
wasn't obvious from the symptom, or where a first fix attempt turned out to be wrong/incomplete
and a second pass was needed. Skip routine bug fixes with an obvious cause.

**Entry format:** one `##` heading (short, symptom-oriented), then:
- **What happened** — the observable symptom, in one or two sentences.
- **Root cause** — what was actually wrong, not just what the symptom looked like.
- **Lesson** — the generalizable takeaway, phrased so it transfers to a different bug shape.

Keep entries short (a paragraph or two). Link to a dedicated doc (like
`null-query-refusal-bug.md`) if the full investigation deserves more space than fits here.

---

## An eval metric can be tautologically green

**What happened.** A generation eval run reported `grounded_claim_rate: 1.0` and
`citation_validity: 1.0` — both look perfect — while `refusal_accuracy` on one bucket
(`null_query`) was 0.233. The pipeline was dumping raw, often irrelevant, retrieved evidence as
if it were an answer, and the eval never noticed because it wasn't checking.

**Root cause.** The fallback path that produced these "answers" (`_numbered_fallback` in
`rag_pipeline.py`) computed `grounded_claims = claims` and `valid_citation_references =
references` by construction, not by verifying anything — so any metric that compares those two
numbers to each other is comparing a number to itself. Full writeup: `null-query-refusal-bug.md`.

**Lesson.** Before trusting a metric that looks clean, check whether it's actually comparing two
independently-derived values, or comparing a value to something derived from the same value. A
metric that can't mathematically fail for a whole code path is not measuring that code path.

## Fixing one bug can be masking a second, worse one

**What happened.** After fixing the `null_query` refusal bug, `null_query` refusal accuracy
jumped from 0.233 to 0.967 — but `inference_query` collapsed from 0.988 to 0.025 in the same run.
The instinct was to treat this as a regression introduced by the fix.

**Root cause.** It wasn't a regression — the "0.988" baseline was never real. `inference_query`
was *also* hitting the same unsafe fallback path before the fix, but the fallback's raw-evidence
dump happened to not contain the refusal string, so it scored as "correctly answered" regardless
of whether the dumped evidence was actually relevant. Fixing the fallback didn't break
`inference_query` — it stopped a different, pre-existing failure from being invisible.

**Lesson.** When a fix flips one metric from bad to good and a different metric from good to bad
in the same run, don't assume the fix caused the second regression. Check whether the "good"
baseline was ever a real measurement of correctness, or just a different symptom of the same
underlying bug that happened to read as passing.

## A safety-net fallback needs its own safety net

**What happened.** `_generate`'s LLM-verification step (`_grounded_answer`) is exactly designed
to catch "no real answer exists" and refuse — but when the LLM's structured response merely
failed to *parse* (unrelated to whether an answer exists), the code fell through to a much more
permissive fallback (dump the evidence) instead of the refusal the verification step would have
produced if it had parsed.

**Root cause.** Two different failure conditions ("the model correctly says no" vs. "we couldn't
tell what the model said") were funneled into the same recovery path, and that path's behavior
was tuned for a third, unrelated condition ("no model is configured at all").

**Lesson.** When a function has more than one reason to fall through to a fallback, check that
the fallback's behavior is actually correct for *every* one of those reasons, not just the one it
was originally written for. "We don't know" and "we know the answer is no" should rarely share a
recovery path with "there's nothing to even ask."

## Chasing a model's specific output quirk with regex is a losing game

**What happened.** MiniMax-M3 (via OpenRouter) wraps JSON-mode output in a visible `<think>`
reasoning block. A regex to strip `<think>...</think>` fixed the first observed failure — then
live testing turned up three more shapes for the *identical* prompt at `temperature=0`: an
unclosed think tag with valid JSON after it, a trailing-only fence with no opening one, and
reasoning that quotes the target JSON schema inline before the real answer.

**Root cause.** Even at `temperature=0`, real-world LLM APIs aren't perfectly deterministic about
formatting, and a provider's output wrapping isn't a fixed, well-specified contract — it's
observed behavior that can vary call to call. A regex tuned to one observed shape is
overfit to that one sample.

**Lesson.** For "extract the JSON from this LLM completion" problems, don't pattern-match the
wrapper — parse for the payload directly. Scan for every plausible JSON start (`{`), let a real
JSON decoder (`json.JSONDecoder.raw_decode`) attempt each one, and keep whichever candidate
consumes the most of the text. That one rule handles nested objects (an inner object always ends
before its enclosing one) and inline schema-quoting (the real, final answer ends later in the
text than an example quoted mid-reasoning) without needing to enumerate wrapper shapes at all.
Verify this kind of fix against live model output, not just hand-written test fixtures — the
three additional failure shapes here were only found by re-running the actual model repeatedly,
not by reasoning about it in the abstract.

## A shared token budget starves whichever call needs it most

**What happened.** A 2000-token budget was shared between a model's visible reasoning and its
actual answer. For a 32-evidence-source multi-hop query, the reasoning alone produced 8000+
characters and was cut off mid-sentence before any answer content appeared — deterministically,
every time, for that query shape.

**Root cause.** The token budget was sized for "a short answer," not for "extensive reasoning
plus a short answer" — a reasoning model doesn't trade off internally between the two; it
happily spends the whole budget on reasoning and gets truncated before answering.

**Lesson.** When a model call can produce internal reasoning that isn't the final answer, budget
for reasoning and answer as separate concerns, even if the API only exposes one combined limit.
Look for whether the provider has a first-class control before working around it (OpenRouter has
a `reasoning.exclude`/`reasoning.effort` parameter) — but verify any such control's actual effect
live before relying on it; here, `reasoning.exclude=true` produced an *empty* answer, worse than
the truncation it was meant to fix, and was abandoned in favor of simply raising the budget.

## LangGraph's state schema silently drops fields it doesn't know about

**What happened.** A new field (`answer_source`) was added to a graph node's return dict and
covered by a unit test that calls the node function directly — the test passed. Running the
*full* pipeline through the actual graph, the field was missing from the final result.

**Root cause.** `AgentState` (a `TypedDict` in `schemas/domain.py`) is what LangGraph uses to
define its state channels. A node returning a key that isn't declared in that TypedDict has
nowhere to be merged into — LangGraph silently drops it, even though plain Python's `TypedDict`
itself enforces nothing at runtime and the node function's own return value looks completely
correct in isolation.

**Lesson.** For any framework that uses a declared schema to wire data between components
(LangGraph state channels, ORM columns, API response models), a unit test that calls the inner
function directly cannot catch "the schema doesn't know about this field." Only a test that goes
through the framework's actual wiring (here: `RagPipeline.run()`, not `_generate()` directly)
exercises that path. Add a new field to the state to the framework's schema and the code in the
same change, and verify with an end-to-end test, not just a function-level one.

## Local infra needs memory caps sized for the actual data, not left to auto-scale

*Scope note: this lesson comes from a historical dev-only reproduction environment that ran
Weaviate locally to debug a specific issue. Weaviate remains cloud-hosted per AGENTS.md — this
section is about the local-only Neo4j container plus that temporary local Weaviate, not a
supported configuration.*

**What happened.** A generation eval run at concurrency 5 failed almost every case with
`httpx.ReadTimeout` against a locally-provisioned Weaviate — even though a single hand-written
query against the same instance responded in ~1 second moments earlier.

**Root cause.** Neo4j and Weaviate, run via Docker Compose with no memory limits, size their own
heap/cache against the full host memory they can see rather than the actual (small, tens-of-
thousands-of-records) dataset — on a shared 8-core/14GB dev machine already running a browser,
IDE, and the rest of the stack, this pushed the host into heavy swapping, and *that* was the
actual cause of the timeouts, not query complexity or concurrency level.

**Lesson.** Before tuning application-level concurrency to fix timeouts against local
infrastructure, check host memory/swap first (`free -h`, `docker stats`) — a resource-starved
host produces symptoms (intermittent timeouts, unexplained slowness) that look exactly like an
application concurrency problem but aren't fixed by changing concurrency at all. Give local
containers explicit, dataset-sized memory limits rather than letting them auto-scale against
total host memory, especially on a shared dev machine.

## A monitoring/tracing sidecar needs its own realistic memory budget

**What happened.** Phoenix (the tracing sidecar), capped at 900MB after the lesson above, was
OOM-killed (exit 137) by a single bulk query fetching many trace spans.

**Root cause.** Each span in a trace can carry a full prompt/response payload — at a 15,000-token
response budget, that's large, and a "reasonable-sounding" memory cap chosen for light, steady-
state ingestion wasn't sized for a heavier read pattern (paging through accumulated history).

**Lesson.** A memory cap picked to solve one problem (overall host memory pressure) can
under-provision for a different access pattern (bulk reads) that wasn't exercised yet. Size caps
for the heaviest realistic operation, not just steady-state load, and expect to revisit them once
real usage patterns (not just steady ingestion) actually exercise the system.

## An API key's format is a claim, not a fact until tested live

**What happened.** `.env` had `LLM_API_KEY` set to a value shaped like `sk-or-v1-<hex>`, and
`LLM_FLASH_MODEL`/`LLM_PRO_MODEL` were configured for MiniMax's native LiteLLM route. Every model
call failed with a 401 from MiniMax's own API.

**Root cause.** The key was an OpenRouter key, not a native MiniMax key — a coincidence of naming
(and the assumption that "a configured key must be valid for the configured model route") stood
in for verification.

**Lesson.** When a credential/model-route combination fails auth, verify with the most direct
possible test (a raw `curl` to the provider's documented endpoint, exactly matching their auth
header format) before concluding the credential itself is wrong vs. the routing is wrong. Here,
the same key worked perfectly once the model strings were changed to OpenRouter's own routing
(`openrouter/minimax/...` + `LLM_BASE_URL=https://openrouter.ai/api/v1`) — the credential was
fine; the assumption about which API it belonged to wasn't.

## Reloading a window doesn't refresh OS-level group membership

**What happened.** After running `sudo usermod -aG docker $USER`, reloading the IDE window (and
even fully restarting the coding assistant's session) still left `docker` commands failing with
a permission error.

**Root cause.** Supplementary group membership is fixed for a process at the time it (or an
ancestor) starts its login session — recalculated by `login`, `sshd`, `su`, `sudo`, `newgrp`, or
equivalent, not by an application-level "reload." The IDE's extension host process itself needed
to fully restart (not just its window) to inherit the new group.

**Lesson.** A permission fix that depends on group/session state doesn't take effect just because
the *tool* using it restarts — trace which OS process actually needs a fresh login/session, and
confirm with `id` (not just "did the reload finish") before concluding the fix didn't work.
