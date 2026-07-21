from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import httpx

HEADERS = {
    "X-Tenant-ID": "demo",
    "X-User-ID": "latency-evaluator",
    "X-Groups": "research",
}


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(math.ceil(len(ordered) * 0.95) - 1, 0)]


async def run_query(client: httpx.AsyncClient, base_url: str, query: str) -> float:
    started = perf_counter()
    session = await client.post(f"{base_url}/api/v1/chat/sessions", headers=HEADERS, json={})
    session.raise_for_status()
    accepted = await client.post(
        f"{base_url}/api/v1/chat/sessions/{session.json()['id']}/messages",
        headers=HEADERS,
        json={"content": query},
    )
    accepted.raise_for_status()
    events = await client.get(f"{base_url}{accepted.json()['events_url']}", headers=HEADERS)
    events.raise_for_status()
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in events.text.splitlines()
        if line.startswith("data: ")
    ]
    if not any(event.get("type") == "answer" for event in payloads):
        raise RuntimeError("Chat run completed without an answer event")
    return (perf_counter() - started) * 1000


async def benchmark(
    dataset: Path,
    thresholds: Path,
    base_url: str,
    repetitions: int,
    concurrency: int,
    route: str | None = None,
) -> dict:
    cases = json.loads(await asyncio.to_thread(dataset.read_text))
    cases = [case for case in cases if route is None or case["route"] == route]
    limits = json.loads(await asyncio.to_thread(thresholds.read_text))
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=300) as client:
        async def measured(case: dict) -> tuple[str, float]:
            async with semaphore:
                return case["route"], await run_query(client, base_url, case["query"])

        samples = await asyncio.gather(
            *(measured(case) for case in cases for _ in range(repetitions))
        )

    grouped: dict[str, list[float]] = defaultdict(list)
    for route, elapsed in samples:
        grouped[route].append(round(elapsed, 2))
    threshold_keys = {
        "direct": "direct_p95_ms",
        "synthesis": "synthesis_p95_ms",
        "complex": "complex_p95_ms",
    }
    routes = {}
    for route, values in grouped.items():
        threshold = limits[threshold_keys[route]]
        p95 = percentile_95(values)
        routes[route] = {
            "samples_ms": values,
            "p95_ms": p95,
            "threshold_ms": threshold,
            "passed": p95 <= threshold,
        }
    return {"repetitions": repetitions, "concurrency": concurrency, "routes": routes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:3000")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--route", choices=("direct", "synthesis", "complex"))
    parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).with_name("latency_dataset.json")
    )
    parser.add_argument(
        "--thresholds", type=Path, default=Path(__file__).with_name("thresholds.json")
    )
    arguments = parser.parse_args()
    report = asyncio.run(
        benchmark(
            arguments.dataset,
            arguments.thresholds,
            arguments.base_url,
            arguments.repetitions,
            arguments.concurrency,
            arguments.route,
        )
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(any(not item["passed"] for item in report["routes"].values()))
