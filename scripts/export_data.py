"""Export Postgres, Neo4j, and Weaviate data so this stack can be respawned elsewhere.

Object storage (MinIO/S3) is out of scope here -- copy that bucket separately
if raw documents need to move too.
"""

import argparse
import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import httpx
from app.core.config import get_settings
from app.services.ingestion import WeaviateIndexer
from neo4j import AsyncGraphDatabase

# label -> properties that uniquely identify a node, matching the MERGE keys
# graph_index.py already uses.
NEO4J_NODE_KEYS = {
    "Document": ("tenant_id", "id"),
    "Entity": ("tenant_id", "key"),
    "Statement": ("tenant_id", "id"),
}
NEO4J_RELATIONSHIPS = [
    ("SUBJECT", "Statement", "Entity"),
    ("OBJECT", "Statement", "Entity"),
    ("SUPPORTED_BY", "Statement", "Document"),
    ("SYNONYM_OF", "Entity", "Entity"),
]


def export_postgres(database_url: str, out_dir: Path) -> None:
    if not shutil.which("pg_dump"):
        raise RuntimeError("pg_dump not found on PATH; install the postgresql client tools")
    subprocess.run(
        ["pg_dump", database_url, "-Fc", "-f", str(out_dir / "postgres.dump")], check=True
    )


async def export_neo4j(uri: str, user: str, password: str, out_dir: Path) -> None:
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        nodes = []
        for label in NEO4J_NODE_KEYS:
            result = await driver.execute_query(f"MATCH (n:{label}) RETURN properties(n) AS props")
            nodes.extend({"label": label, "props": record["props"]} for record in result.records)
        rels = []
        for rel_type, start_label, end_label in NEO4J_RELATIONSHIPS:
            # Only the identifying keys travel here, not full node properties --
            # entities carry an embedding vector and pulling it once per edge
            # (tens of thousands of Statement->Entity edges) made this query
            # take minutes instead of seconds.
            start_proj = ", ".join(f"{key}: a.{key}" for key in NEO4J_NODE_KEYS[start_label])
            end_proj = ", ".join(f"{key}: b.{key}" for key in NEO4J_NODE_KEYS[end_label])
            result = await driver.execute_query(
                f"MATCH (a:{start_label})-[r:{rel_type}]->(b:{end_label}) "
                f"RETURN {{{start_proj}}} AS start, {{{end_proj}}} AS end, properties(r) AS props"
            )
            rels.extend(
                {
                    "type": rel_type,
                    "start_label": start_label,
                    "end_label": end_label,
                    "start": record["start"],
                    "end": record["end"],
                    "props": record["props"],
                }
                for record in result.records
            )
        (out_dir / "neo4j.json").write_text(json.dumps({"nodes": nodes, "rels": rels}))
        print(f"neo4j: {len(nodes)} nodes, {len(rels)} relationships")
    finally:
        await driver.close()


async def export_weaviate(settings, out_dir: Path) -> None:
    headers = {"Authorization": f"Bearer {settings.weaviate_api_key}"}
    url = settings.weaviate_url.rstrip("/")
    collection = settings.weaviate_collection
    fields = " ".join(prop["name"] for prop in WeaviateIndexer._schema_properties())
    count, after = 0, ""
    handle = (out_dir / "weaviate.jsonl").open("w")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                cursor = f',after:"{after}"' if after else ""
                query = (
                    f"query Export {{ Get {{ {collection}(limit:100{cursor}) "
                    f"{{ {fields} _additional {{ id vectors {{ default }} }} }} }} }}"
                )
                response = await client.post(
                    f"{url}/v1/graphql", headers=headers, json={"query": query}
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("errors"):
                    raise RuntimeError(payload["errors"][0]["message"])
                rows = payload.get("data", {}).get("Get", {}).get(collection, [])
                if not rows:
                    break
                after = rows[-1]["_additional"]["id"]
                for row in rows:
                    additional = row.pop("_additional")
                    record = {
                        "id": additional["id"],
                        "vector": additional["vectors"]["default"],
                        "properties": row,
                    }
                    handle.write(json.dumps(record) + "\n")
                    count += 1
                if len(rows) < 100:
                    break
    finally:
        handle.close()
    print(f"weaviate: {count} objects")


async def main(out_dir: Path, only: set[str]) -> None:
    settings = get_settings()
    if "postgres" in only:
        export_postgres(settings.database_url, out_dir)
        print("postgres: done")
    if "neo4j" in only:
        await export_neo4j(
            settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, out_dir
        )
    if "weaviate" in only:
        await export_weaviate(settings, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/backup", type=Path)
    parser.add_argument(
        "--only", choices=["postgres", "neo4j", "weaviate"], nargs="+", default=None
    )
    arguments = parser.parse_args()
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(main(arguments.out_dir, set(arguments.only or ["postgres", "neo4j", "weaviate"])))
