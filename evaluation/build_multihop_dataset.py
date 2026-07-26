"""Build MultiHop-RAG-derived eval fixtures for route accuracy, retrieval recall,
and grounded-generation testing.

Source: https://huggingface.co/datasets/yixuantt/MultiHopRAG (ODC-BY licensed).
2,556 queries over 609 news articles, with question_type in
{inference_query, comparison_query, temporal_query, null_query}. The first three
map onto this app's Route enum; null_query ("unanswerable from this corpus") does
NOT map to Route.OUT_OF_SCOPE -- this app's OUT_OF_SCOPE route is a jailbreak/abuse
guard (see adaptive_router.py), not a knowledge-boundary classifier. null_query is
used only for the generation-eval refusal check.

Usage:
    python evaluation/build_multihop_dataset.py build          # sampled fixtures (fast dev loop)
    python evaluation/build_multihop_dataset.py build --full   # full-scale fixtures (~2556 queries)
    python evaluation/build_multihop_dataset.py ingest [--manifest PATH] [--base-url URL]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

HF_BASE = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main"
ROUTE_BY_TYPE = {
    "inference_query": "multi_hop",
    "comparison_query": "comparison",
    "temporal_query": "temporal_causal",
}
RETRIEVAL_SAMPLE_PER_TYPE = 10
GOLDEN_SAMPLE_PER_TYPE = 40
GENERATION_ANSWERABLE_PER_TYPE = 3
GENERATION_NULL_SAMPLE = 10
SAMPLED_MAX_EVIDENCE = 4
SEED = 42

HERE = Path(__file__).parent
CACHE_DIR = HERE / "multihop_cache"
CORPUS_DIR = HERE / "multihop_corpus"


def _download(name: str) -> list[dict]:
    cache_path = CACHE_DIR / name
    if not cache_path.exists():
        CACHE_DIR.mkdir(exist_ok=True)
        response = httpx.get(f"{HF_BASE}/{name}", follow_redirects=True, timeout=120)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
    return json.loads(cache_path.read_text())


def _safe_filename(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")[:80]


def _select(pool: list[dict], rng: random.Random, sample_per_type: int | None) -> list[dict]:
    return pool if sample_per_type is None else rng.sample(pool, min(sample_per_type, len(pool)))


def _build_golden(
    queries: list[dict], rng: random.Random, sample_per_type: int | None
) -> list[dict]:
    golden = []
    for qtype, route in ROUTE_BY_TYPE.items():
        pool = [item for item in queries if item["question_type"] == qtype]
        for item in _select(pool, rng, sample_per_type):
            golden.append({"query": item["query"], "route": route, "question_type": qtype})
    return golden


def _build_retrieval(
    queries: list[dict],
    corpus_titles: set[str],
    rng: random.Random,
    sample_per_type: int | None,
    max_evidence: int | None,
) -> list[dict]:
    retrieval_cases = []
    for qtype, route in ROUTE_BY_TYPE.items():
        pool = [
            item
            for item in queries
            if item["question_type"] == qtype
            and len(item["evidence_list"]) >= 2
            and (max_evidence is None or len(item["evidence_list"]) <= max_evidence)
            and all(e["title"] in corpus_titles for e in item["evidence_list"])
        ]
        for item in _select(pool, rng, sample_per_type):
            titles = sorted({e["title"] for e in item["evidence_list"]})
            retrieval_cases.append(
                {
                    "name": item["query"][:80],
                    "query": item["query"],
                    "route": route,
                    "question_type": qtype,
                    "evidence_count": len(item["evidence_list"]),
                    "tenant_id": "eval",
                    "groups": ["research"],
                    "expected_document_titles": titles,
                    "expected_answer": item["answer"],
                }
            )
    return retrieval_cases


def _build_generation(
    queries: list[dict],
    retrieval_cases: list[dict],
    rng: random.Random,
    answerable_per_type: int | None,
    null_sample: int | None,
) -> list[dict]:
    null_pool = [item for item in queries if item["question_type"] == "null_query"]
    answerable_cases = (
        retrieval_cases
        if answerable_per_type is None
        else retrieval_cases[: answerable_per_type * len(ROUTE_BY_TYPE)]
    )
    generation_cases = [{**case, "expect_refusal": False} for case in answerable_cases]
    generation_cases += [
        {
            "name": item["query"][:80],
            "query": item["query"],
            "route": None,
            "question_type": "null_query",
            "tenant_id": "eval",
            "groups": ["research"],
            "expected_document_titles": [],
            "expect_refusal": True,
        }
        for item in _select(null_pool, rng, null_sample)
    ]
    return generation_cases


def _write_corpus(
    titles: set[str], corpus_by_title: dict[str, dict], manifest_name: str
) -> int:
    CORPUS_DIR.mkdir(exist_ok=True)
    manifest = []
    for title in sorted(titles):
        doc = corpus_by_title[title]
        path = CORPUS_DIR / f"{_safe_filename(title)}.md"
        path.write_text(f"# {doc['title']}\n\n{doc['body']}\n")
        manifest.append({"title": doc["title"], "path": path.name})
    (CORPUS_DIR / manifest_name).write_text(json.dumps(manifest, indent=2))
    return len(titles)


def build(
    full: bool = False,
    retrieval_sample: int | None = RETRIEVAL_SAMPLE_PER_TYPE,
    golden_sample: int | None = GOLDEN_SAMPLE_PER_TYPE,
    gen_answerable: int | None = GENERATION_ANSWERABLE_PER_TYPE,
    gen_null: int | None = GENERATION_NULL_SAMPLE,
) -> None:
    queries = _download("MultiHopRAG.json")
    corpus = _download("corpus.json")
    corpus_titles = {item["title"] for item in corpus}
    corpus_by_title = {item["title"]: item for item in corpus}
    rng = random.Random(SEED)

    if full:
        retrieval_sample = golden_sample = gen_answerable = gen_null = None
    max_evidence = None if full else SAMPLED_MAX_EVIDENCE
    suffix = ".full" if full else ""

    golden = _build_golden(queries, rng, golden_sample)
    (HERE / f"multihop_golden_dataset{suffix}.json").write_text(json.dumps(golden, indent=2))

    retrieval_cases = _build_retrieval(
        queries, corpus_titles, rng, retrieval_sample, max_evidence
    )
    (HERE / f"multihop_retrieval_dataset{suffix}.json").write_text(
        json.dumps(retrieval_cases, indent=2)
    )

    generation_cases = _build_generation(queries, retrieval_cases, rng, gen_answerable, gen_null)
    (HERE / f"generation_dataset{suffix}.json").write_text(json.dumps(generation_cases, indent=2))

    required_titles = (
        corpus_titles
        if full
        else {title for case in retrieval_cases for title in case["expected_document_titles"]}
    )
    article_count = _write_corpus(required_titles, corpus_by_title, f"manifest{suffix}.json")

    print(f"golden: {len(golden)} cases (route accuracy, no ingestion needed)")
    print(f"retrieval: {len(retrieval_cases)} cases, {article_count} articles to ingest")
    print(f"generation: {len(generation_cases)} cases")


async def ingest(base_url: str, manifest: Path, timeout_seconds: int = 900) -> None:
    manifest_entries = json.loads(await asyncio.to_thread(manifest.read_text))
    headers = {"X-Tenant-ID": "eval", "X-User-ID": "eval-builder", "X-Groups": "research"}

    def _is_rate_limited(exc: BaseException) -> bool:
        return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429

    async with httpx.AsyncClient(timeout=30) as client:
        existing = await client.get(f"{base_url}/api/v1/documents", headers=headers)
        existing.raise_for_status()
        existing_titles = {doc["title"] for doc in existing.json()}
        to_upload = [e for e in manifest_entries if e["title"] not in existing_titles]
        skipped = len(manifest_entries) - len(to_upload)
        if skipped:
            print(f"skipping {skipped} already-ingested titles")

        # The API's rate limit is a per-tenant-per-wallclock-minute Redis
        # counter (auth.py enforce_rate_limit) that increments on every
        # request, including ones that get 429'd -- so retries fired
        # concurrently only dig the same minute's budget deeper and never let
        # it recover. Upload sequentially, one request in flight at a time,
        # paced comfortably under requests_per_minute; the retry below is a
        # backstop for a budget already exhausted by other traffic on this
        # tenant, not a way to push through this loop's own request rate.
        @retry(
            retry=retry_if_exception(_is_rate_limited),
            wait=wait_random_exponential(multiplier=1, max=90),
            stop=stop_after_attempt(8),
        )
        async def upload(entry: dict) -> str:
            content = (manifest.parent / entry["path"]).read_bytes()
            response = await client.post(
                f"{base_url}/api/v1/documents",
                headers=headers,
                files={"file": (entry["path"], content, "text/markdown")},
                data={"title": entry["title"], "acl_groups": "research"},
            )
            response.raise_for_status()
            return response.json()["ingestion_job"]["id"]

        job_ids = []
        for entry in to_upload:
            job_ids.append(await upload(entry))
            await asyncio.sleep(0.6)
        print(f"submitted {len(job_ids)} ingestion jobs")
        pending = set(job_ids)
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while pending and asyncio.get_event_loop().time() < deadline:
            for job_id in list(pending):
                # Stay well under requests_per_minute (shared by all polls in this
                # sweep, same tenant): space individual polls out instead of
                # firing len(pending) requests in the same instant. At full scale
                # (609 jobs) one sweep costs ~609*0.5s=~305s, so timeout_seconds
                # needs a large default (see --timeout-seconds) -- ~3600s gives
                # ~10 sweeps of headroom for jobs to settle at different rates.
                await asyncio.sleep(0.5)
                response = await client.get(
                    f"{base_url}/api/v1/ingestion-jobs/{job_id}", headers=headers
                )
                if response.status_code == 429:
                    continue
                response.raise_for_status()
                status = response.json()["status"]
                if status in {"complete", "failed"}:
                    pending.discard(job_id)
                    if status == "failed":
                        print(f"WARNING: job {job_id} failed")
            if pending:
                print(f"{len(pending)} jobs still pending...")
                await asyncio.sleep(5)
        if pending:
            raise TimeoutError(
                f"{len(pending)} ingestion jobs still pending after {timeout_seconds}s"
            )
        print("all ingestion jobs settled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "ingest"])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--full", action="store_true", help="Build/ingest at full dataset scale")
    parser.add_argument("--retrieval-sample-per-type", type=int, default=RETRIEVAL_SAMPLE_PER_TYPE)
    parser.add_argument("--golden-sample-per-type", type=int, default=GOLDEN_SAMPLE_PER_TYPE)
    parser.add_argument(
        "--generation-answerable-per-type", type=int, default=GENERATION_ANSWERABLE_PER_TYPE
    )
    parser.add_argument("--generation-null-sample", type=int, default=GENERATION_NULL_SAMPLE)
    parser.add_argument("--manifest", type=Path, default=CORPUS_DIR / "manifest.json")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    arguments = parser.parse_args()
    if arguments.action == "build":
        build(
            full=arguments.full,
            retrieval_sample=arguments.retrieval_sample_per_type,
            golden_sample=arguments.golden_sample_per_type,
            gen_answerable=arguments.generation_answerable_per_type,
            gen_null=arguments.generation_null_sample,
        )
    else:
        asyncio.run(ingest(arguments.base_url, arguments.manifest, arguments.timeout_seconds))
