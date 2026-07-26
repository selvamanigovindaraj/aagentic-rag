"""Label legacy Weaviate/Neo4j records before enabling index-version filters."""

import argparse
import asyncio
import json

import httpx
from app.core.config import get_settings
from neo4j import AsyncGraphDatabase


async def backfill_weaviate(version: str, collection_override: str | None = None) -> int:
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.weaviate_api_key}"}
    url = settings.weaviate_url.rstrip("/")
    collection = collection_override or settings.weaviate_collection
    count, after = 0, ""
    async with httpx.AsyncClient(timeout=30) as client:
        schema = await client.get(f"{url}/v1/schema/{collection}", headers=headers)
        schema.raise_for_status()
        if "indexVersion" not in {item["name"] for item in schema.json()["properties"]}:
            response = await client.post(
                f"{url}/v1/schema/{collection}/properties",
                headers=headers,
                json={"name": "indexVersion", "dataType": ["text"]},
            )
            response.raise_for_status()
        while True:
            cursor = f',after:"{after}"' if after else ""
            query = """query Backfill { Get { %s(limit:100%s) {
              indexVersion _additional { id }
            }}}""" % (collection, cursor)  # noqa: UP031
            response = await client.post(
                f"{url}/v1/graphql", headers=headers, json={"query": query}
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                raise RuntimeError(payload["errors"][0]["message"])
            rows = payload.get("data", {}).get("Get", {}).get(collection, [])
            for row in rows:
                if not row.get("indexVersion"):
                    patched = await client.patch(
                        f"{url}/v1/objects/{collection}/{row['_additional']['id']}",
                        headers=headers,
                        json={"class": collection, "properties": {"indexVersion": version}},
                    )
                    patched.raise_for_status()
                    count += 1
            if len(rows) < 100:
                return count
            after = rows[-1]["_additional"]["id"]


async def backfill_neo4j(version: str) -> tuple[int, int]:
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        statement_result = await driver.execute_query(
            """MATCH (s:Statement) WHERE s.index_version IS NULL
            SET s.index_version=$version RETURN count(s) AS count""",
            version=version,
        )
        synonym_result = await driver.execute_query(
            """MATCH ()-[r:SYNONYM_OF]->() WHERE r.index_version IS NULL
            SET r.index_version=$version RETURN count(r) AS count""",
            version=version,
        )
        return statement_result[0][0]["count"], synonym_result[0][0]["count"]
    finally:
        await driver.close()


async def main(version: str, collection: str | None) -> None:
    vectors, (statements, synonyms) = await asyncio.gather(
        backfill_weaviate(version, collection), backfill_neo4j(version)
    )
    print(json.dumps({"weaviate": vectors, "statements": statements, "synonyms": synonyms}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--collection")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.version, arguments.collection))
