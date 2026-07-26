"""Import a backup made by export_data.py into a fresh Postgres/Neo4j/Weaviate stack."""

import argparse
import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import httpx
from app.core.config import get_settings
from app.services.ingestion import WeaviateIndexer
from export_data import NEO4J_NODE_KEYS
from neo4j import AsyncGraphDatabase

BATCH_SIZE = 1000


def _batches(rows: list) -> list[list]:
    return [rows[i : i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]


def import_postgres(database_url: str, out_dir: Path) -> None:
    if not shutil.which("pg_restore"):
        raise RuntimeError("pg_restore not found on PATH; install the postgresql client tools")
    subprocess.run(
        [
            "pg_restore",
            "--clean",
            "--if-exists",
            "--no-owner",
            "--no-privileges",
            "-d",
            database_url,
            str(out_dir / "postgres.dump"),
        ],
        check=True,
    )


async def import_neo4j(uri: str, user: str, password: str, out_dir: Path) -> None:
    payload = json.loads((out_dir / "neo4j.json").read_text())
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        # The app never creates a Neo4j index (small per-document MERGEs never
        # needed one), so a bulk MERGE across tens of thousands of rows
        # degrades to a full label scan per row -- minutes instead of seconds.
        for label, keys in NEO4J_NODE_KEYS.items():
            props = ", ".join(f"n.{key}" for key in keys)
            await driver.execute_query(
                f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON ({props})"
            )
        await driver.execute_query("CALL db.awaitIndexes(300)")
        for label, keys in NEO4J_NODE_KEYS.items():
            rows = [node["props"] for node in payload["nodes"] if node["label"] == label]
            if not rows:
                continue
            key_pattern = ", ".join(f"{key}: row.{key}" for key in keys)
            for batch in _batches(rows):
                await driver.execute_query(
                    f"UNWIND $rows AS row MERGE (n:{label} {{{key_pattern}}}) SET n += row",
                    rows=batch,
                )
        grouped: dict[tuple[str, str, str], list[dict]] = {}
        for rel in payload["rels"]:
            grouped.setdefault((rel["type"], rel["start_label"], rel["end_label"]), []).append(rel)
        for (rel_type, start_label, end_label), rows in grouped.items():
            start_keys = NEO4J_NODE_KEYS[start_label]
            end_keys = NEO4J_NODE_KEYS[end_label]
            start_pattern = ", ".join(f"{key}: row.start.{key}" for key in start_keys)
            end_pattern = ", ".join(f"{key}: row.end.{key}" for key in end_keys)
            for batch in _batches(rows):
                await driver.execute_query(
                    f"""
                    UNWIND $rows AS row
                    MATCH (a:{start_label} {{{start_pattern}}}), (b:{end_label} {{{end_pattern}}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r += row.props
                    """,
                    rows=batch,
                )
        print(f"neo4j: {len(payload['nodes'])} nodes, {len(payload['rels'])} relationships")
    finally:
        await driver.close()


async def import_weaviate(settings, out_dir: Path) -> None:
    headers = {"Authorization": f"Bearer {settings.weaviate_api_key}"}
    url = settings.weaviate_url.rstrip("/")
    collection = settings.weaviate_collection
    async with httpx.AsyncClient(timeout=30) as client:
        indexer = WeaviateIndexer(
            url, settings.weaviate_api_key, None, None, client, collection, settings.index_version
        )
        await indexer._ensure_schema()
        lines = (out_dir / "weaviate.jsonl").read_text().splitlines()
        count = 0
        for start in range(0, len(lines), 100):
            batch = [json.loads(line) for line in lines[start : start + 100]]
            objects = [
                {
                    "class": collection,
                    "id": item["id"],
                    "vector": item["vector"],
                    "properties": item["properties"],
                }
                for item in batch
            ]
            response = await client.post(
                f"{url}/v1/batch/objects", headers=headers, json={"objects": objects}
            )
            response.raise_for_status()
            count += len(objects)
    print(f"weaviate: {count} objects")


async def main(out_dir: Path, only: set[str]) -> None:
    settings = get_settings()
    if "postgres" in only:
        import_postgres(settings.database_url, out_dir)
        print("postgres: done")
    if "neo4j" in only:
        await import_neo4j(
            settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, out_dir
        )
    if "weaviate" in only:
        await import_weaviate(settings, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/backup", type=Path)
    parser.add_argument(
        "--only", choices=["postgres", "neo4j", "weaviate"], nargs="+", default=None
    )
    arguments = parser.parse_args()
    asyncio.run(main(arguments.out_dir, set(arguments.only or ["postgres", "neo4j", "weaviate"])))
