"""Runs Phoenix's LLM-graded evaluators (hallucination, QA correctness,
retrieval relevance) against `generate` spans already captured in Phoenix,
using phoenix.evals end-to-end -- no hand-rolled grading logic. The judge
model reuses this app's own LiteLLM setup via phoenix.evals' native litellm
provider adapter (same MINIMAX_API_KEY auto-discovery LiteLLMGateway relies
on), scored, then logged back onto the source spans as annotations.

Usage:
    python evaluation/phoenix_evals.py [--project agentic-rag] [--limit 1000]
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from app.core.config import Settings
from phoenix.client import Client
from phoenix.evals import LLM, create_classifier, evaluate_dataframe

HALLUCINATION_TEMPLATE = """You are evaluating whether an answer is factually
grounded in the given reference context.

[Query]: {query}
[Reference]: {reference}
[Answer]: {response}

Is the answer fully supported by the reference, with no fabricated claims?
Answer "factual" if every claim is supported, "hallucinated" if the answer
contains claims not supported by the reference."""

QA_CORRECTNESS_TEMPLATE = """You are evaluating whether an answer correctly
and completely answers the question, using only the reference context.

[Query]: {query}
[Reference]: {reference}
[Answer]: {response}

Answer "correct" if the answer correctly addresses the query using the
reference, "incorrect" otherwise."""

RETRIEVAL_RELEVANCE_TEMPLATE = """You are evaluating whether the retrieved
reference context is relevant to answering the query.

[Query]: {query}
[Reference]: {reference}

Answer "relevant" if the reference contains information useful for answering
the query, "irrelevant" if it does not."""


def _load_generate_spans(client: Client, project_name: str, limit: int) -> pd.DataFrame:
    spans = client.spans.get_spans_dataframe(project_name=project_name, limit=limit)
    generate_spans = spans[spans["name"] == "generate"]
    records = []
    for span_id, row in generate_spans.iterrows():
        input_data = json.loads(row["attributes.input.value"])
        output_data = json.loads(row["attributes.output.value"])
        evidence = input_data.get("accepted_evidence") or []
        if not evidence or not output_data.get("answer"):
            continue
        records.append(
            {
                "context.span_id": span_id,
                "query": input_data["query"],
                "reference": "\n\n".join(item["text"] for item in evidence),
                "response": output_data["answer"],
            }
        )
    return pd.DataFrame(records)


def build_evaluators(llm: LLM) -> list:
    # choices as {label: (score, description)} tuples forces
    # ObjectGenerationMethod.TOOL_CALLING (native function-calling) instead of
    # AUTO's raw JSON-mode text parsing -- MiniMax wraps JSON-mode output in a
    # ```json fence despite response_format=json_object (the same quirk
    # LiteLLMGateway._strip_json_fence works around for this app's own calls;
    # phoenix.evals' separate LiteLLM client has no such workaround, so
    # tool-calling sidesteps the issue instead of reimplementing that fix).
    return [
        create_classifier(
            name="hallucination",
            llm=llm,
            prompt_template=HALLUCINATION_TEMPLATE,
            choices={
                "factual": (1, "every claim is supported by the reference"),
                "hallucinated": (0, "the answer contains unsupported claims"),
            },
        ),
        create_classifier(
            name="qa_correctness",
            llm=llm,
            prompt_template=QA_CORRECTNESS_TEMPLATE,
            choices={
                "correct": (1, "the answer correctly addresses the query"),
                "incorrect": (0, "the answer does not correctly address the query"),
            },
        ),
        create_classifier(
            name="retrieval_relevance",
            llm=llm,
            prompt_template=RETRIEVAL_RELEVANCE_TEMPLATE,
            choices={
                "relevant": (1, "the reference is useful for answering the query"),
                "irrelevant": (0, "the reference is not useful for answering the query"),
            },
        ),
    ]


def _annotations_frame(results: pd.DataFrame, evaluators: list) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        for evaluator in evaluators:
            raw = row[f"{evaluator.name}_score"]
            if not isinstance(raw, str):
                continue  # judge call exhausted retries for this row; skip, don't crash the batch
            scores = json.loads(raw)
            for item in scores:
                rows.append(
                    {
                        "span_id": row["context.span_id"],
                        "name": item["name"],
                        "label": item.get("label"),
                        "score": item.get("score"),
                        "explanation": item.get("explanation"),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["span_id", "name", "label", "score", "explanation"]).set_index(
            "span_id"
        )
    return pd.DataFrame(rows).set_index("span_id")


def run(base_url: str, project_name: str, limit: int) -> dict:
    """Some rows may fail even with TOOL_CALLING forced (phoenix.evals'
    litellm adapter sets tool_choice explicitly) -- observed live: MiniMax
    occasionally returns tool_calls=None despite the forced tool_choice,
    same class of provider quirk as the JSON-fence-wrapping and
    unsupported-"thinking"-param issues already documented in
    LiteLLMGateway. _annotations_frame skips those rows rather than losing
    the whole batch."""
    settings = Settings()
    client = Client(base_url=base_url)
    dataframe = _load_generate_spans(client, project_name, limit)
    if dataframe.empty:
        print("no generate spans with evidence+answer found -- nothing to evaluate")
        return {}
    llm = LLM(provider="litellm", model=settings.llm_pro_model)
    evaluators = build_evaluators(llm)
    results = evaluate_dataframe(dataframe=dataframe, evaluators=evaluators)
    annotations = _annotations_frame(results, evaluators)
    if annotations.empty:
        print(
            "all evaluator calls failed this run (see MiniMax tool-calling reliability "
            "note above) -- nothing to log"
        )
        return {}
    client.spans.log_span_annotations_dataframe(dataframe=annotations, annotator_kind="LLM")
    return {
        evaluator.name: annotations.loc[annotations["name"] == evaluator.name, "score"].mean()
        for evaluator in evaluators
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:6006")
    parser.add_argument("--project", default="agentic-rag")
    parser.add_argument("--limit", type=int, default=1000)
    arguments = parser.parse_args()
    summary = run(arguments.base_url, arguments.project, arguments.limit)
    print(json.dumps(summary, indent=2))
