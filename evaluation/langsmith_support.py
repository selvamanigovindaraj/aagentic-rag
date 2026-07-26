from __future__ import annotations

from langsmith import Client


def ensure_dataset(
    client: Client, name: str, examples: list[dict], *, description: str = ""
) -> str:
    """Idempotently sync local golden-dataset examples into a LangSmith dataset,
    keyed by name; only adds examples missing by their `name` field, never
    duplicates or overwrites what's already there. Returns the dataset id."""
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(dataset_name=name, description=description)
    dataset = client.read_dataset(dataset_name=name)
    existing = {
        example.inputs.get("name"): example
        for example in client.list_examples(dataset_id=dataset.id)
    }
    missing = [item for item in examples if item["name"] not in existing]
    if missing:
        client.create_examples(
            dataset_id=dataset.id,
            examples=[{"inputs": item, "outputs": {}} for item in missing],
        )
    changed = [
        item
        for item in examples
        if item["name"] in existing and existing[item["name"]].inputs != item
    ]
    if changed:
        client.update_examples(
            dataset_id=dataset.id,
            updates=[
                {"id": existing[item["name"]].id, "inputs": item} for item in changed
            ],
        )
    return str(dataset.id)
