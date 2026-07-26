from __future__ import annotations

import argparse
import asyncio
import time

import httpx


async def ingest(client: httpx.AsyncClient, base_url: str, index: int) -> str:
    response = await client.post(
        f"{base_url}/api/v1/documents",
        headers={
            "X-Tenant-ID": "scale-eval",
            "X-User-ID": "load-test",
            "X-Groups": "evaluation",
        },
        data={"title": f"Scale document {index}", "acl_groups": "evaluation"},
        files={
            "file": (
                f"document-{index}.txt",
                f"Document {index}\n\nUnique identifier SCALE-{index:06d}. " * 20,
                "text/plain",
            )
        },
    )
    response.raise_for_status()
    return response.json()["ingestion_job"]["id"]


async def run(count: int, concurrency: int, base_url: str) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as client:

        async def limited(index: int) -> str:
            async with semaphore:
                return await ingest(client, base_url, index)

        jobs = await asyncio.gather(*(limited(index) for index in range(count)))
        pending = set(jobs)
        failed = 0
        headers = {
            "X-Tenant-ID": "scale-eval",
            "X-User-ID": "load-test",
            "X-Groups": "evaluation",
        }
        while pending:
            sample = list(pending)[:concurrency]
            responses = await asyncio.gather(
                *(
                    client.get(f"{base_url}/api/v1/ingestion-jobs/{job}", headers=headers)
                    for job in sample
                )
            )
            for job, response in zip(sample, responses, strict=True):
                response.raise_for_status()
                status = response.json()["status"]
                if status in {"complete", "failed"}:
                    pending.remove(job)
                    failed += status == "failed"
            if pending:
                await asyncio.sleep(1)
    elapsed = time.perf_counter() - started
    print(
        f"completed={count - failed} failed={failed} seconds={elapsed:.2f} "
        f"documents_per_second={count / elapsed:.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--base-url", default="http://localhost:8000")
    arguments = parser.parse_args()
    asyncio.run(run(arguments.count, arguments.concurrency, arguments.base_url))
