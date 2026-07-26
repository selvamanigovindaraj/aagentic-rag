"""Logs an already-computed eval report (retrieval_eval.py / generation_eval.py
/ offline_eval.py JSON output) into Phoenix as a tracked experiment, using
phoenix.client's low-level experiment primitives (create + log_run +
log_evaluation) -- the result is already computed by our own eval scripts, not
re-run through Phoenix's own task executor, so this only records it.

Usage:
    python evaluation/phoenix_experiments.py retrieval \
        --report evaluation/full_retrieval_report.json
    python evaluation/phoenix_experiments.py generation \
        --report evaluation/full_generation_report.json
    python evaluation/phoenix_experiments.py route \
        --report evaluation/full_route_report.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from phoenix.client import Client

# (dataset input keys, dataset output keys, the case field to log as the
# CODE-annotator evaluation score, an optional field to log as its label)
REPORT_SPECS = {
    "retrieval": (["query"], ["retrieved_document_titles"], "recall_at_20", "question_type"),
    "generation": (
        ["name"],
        ["refused", "grounded", "citations_valid"],
        "refusal_correct",
        "question_type",
    ),
    "route": (["query"], ["actual"], "correct", "expected"),
}


def _dataset_examples(
    cases: list[dict], input_keys: list[str], output_keys: list[str]
) -> list[dict]:
    return [
        {
            "input": {key: case.get(key) for key in input_keys},
            "output": {key: case.get(key) for key in output_keys},
            "metadata": {"name": case.get("name") or str(case.get("query", ""))[:80]},
        }
        for case in cases
    ]


def log_experiment(
    client: Client,
    dataset_name: str,
    experiment_name: str,
    cases: list[dict],
    input_keys: list[str],
    output_keys: list[str],
    score_key: str,
    label_key: str | None,
) -> str:
    examples = _dataset_examples(cases, input_keys, output_keys)
    dataset = client.datasets.create_dataset(name=dataset_name, examples=examples)
    experiment = client.experiments.create(dataset_id=dataset.id, experiment_name=experiment_name)
    for example, case in zip(dataset.examples, cases, strict=False):
        now = datetime.now(UTC)
        run = client.experiments.log_run(
            experiment_id=experiment["id"],
            dataset_example_id=example["id"],
            output={key: case.get(key) for key in output_keys},
            start_time=now,
            end_time=now,
        )
        score = case.get(score_key)
        if score is None:
            continue
        client.experiments.log_evaluation(
            experiment_run_id=run["id"],
            name=score_key,
            annotator_kind="CODE",
            score=float(score),
            label=str(case.get(label_key)) if label_key else None,
        )
    return experiment["id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=list(REPORT_SPECS))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:6006")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--experiment-name", default=None)
    arguments = parser.parse_args()

    report = json.loads(arguments.report.read_text())
    cases = report["cases"]
    input_keys, output_keys, score_key, label_key = REPORT_SPECS[arguments.kind]
    experiment_id = log_experiment(
        Client(base_url=arguments.base_url),
        arguments.dataset_name or f"{arguments.kind}-eval",
        arguments.experiment_name or f"{arguments.kind}-eval-{arguments.report.stem}",
        cases,
        input_keys,
        output_keys,
        score_key,
        label_key,
    )
    print(f"logged {len(cases)} cases as Phoenix experiment {experiment_id}")
