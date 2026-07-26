import argparse
import asyncio
import json
from pathlib import Path

from app.agents.adaptive_router import classify


def route_accuracy(dataset: Path) -> float:
    examples = json.loads(dataset.read_text())
    correct = sum(classify(item["query"]) == item["route"] for item in examples)
    return correct / len(examples)


def _breakdown_route_accuracy(results: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for item in results:
        grouped.setdefault(str(item["question_type"]), []).append(item)
    return {
        bucket: {
            "count": len(items),
            "accuracy": sum(i["correct"] for i in items) / len(items),
        }
        for bucket, items in grouped.items()
    }


async def route_accuracy_live(dataset: Path) -> dict:
    """Route accuracy through the real classify path (keyword router + the LLM
    second-pass override), not just the raw keyword function -- this is what
    production actually routes with, and the two can disagree (see
    rag_pipeline.py's RagPipeline._classify: the LLM now always gets a look,
    including the cases the keyword router defaults to `direct`)."""
    from app.components.llm import LiteLLMGateway
    from app.components.retrieval import EmptyRetriever
    from app.core.config import Settings
    from app.services.rag_pipeline import RagPipeline

    examples = json.loads(await asyncio.to_thread(dataset.read_text))
    settings = Settings()
    pipeline = RagPipeline(EmptyRetriever(), settings, LiteLLMGateway(settings))
    results = []
    for item in examples:
        state = await pipeline._classify({"query": item["query"], "model_calls": 0})
        results.append(
            {
                "query": item["query"][:80],
                "question_type": item.get("question_type"),
                "expected": item["route"],
                "actual": str(state["route"]),
                "correct": str(state["route"]) == item["route"],
            }
        )
    accuracy = sum(item["correct"] for item in results) / len(results)
    breakdown = _breakdown_route_accuracy([r for r in results if r["question_type"]])
    return {"route_accuracy_live": accuracy, "by_question_type": breakdown, "cases": results}


if __name__ == "__main__":
    from app.core.tracing import configure_tracing

    configure_tracing()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).with_name("golden_dataset.json")
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Route through the real LLM-augmented classify path instead of the raw keyword router",
    )
    arguments = parser.parse_args()
    if arguments.live:
        report = asyncio.run(route_accuracy_live(arguments.dataset))
        print(json.dumps(report, indent=2))
        accuracy = report["route_accuracy_live"]
    else:
        accuracy = route_accuracy(arguments.dataset)
        print(f"route_accuracy={accuracy:.3f}")
    raise SystemExit(accuracy < arguments.threshold)
