from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from dataclasses import dataclass, replace

import numpy as np
from openai import OpenAIError
from pydantic import BaseModel, Field
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from ..components.llm import ModelGateway
from ..components.voyage import Embedder
from ..prompts.registry import prompt
from .groundedness import is_grounded
from .ingestion import Chunk

logger = logging.getLogger(__name__)

MAX_SUMMARY_CHARACTERS = 4_000


class SummaryOutput(BaseModel):
    summary: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class SummaryNode:
    key: str
    text: str
    level: int
    child_ids: tuple[str, ...]
    page: int | None
    section: str | None
    vector: list[float]
    source_ids: tuple[str, ...]
    is_root: bool = False


async def build_raptor(
    chunks: list[Chunk], models: ModelGateway, embedder: Embedder, *, cluster_size: int = 5
) -> list[SummaryNode]:
    """Build an evidence-linked, bottom-up summary tree using embedding similarity."""
    leaf_texts = [chunk.text for chunk in chunks]
    current = [
        (
            f"chunk-{chunk.index}",
            chunk.text,
            chunk.page,
            chunk.section,
            vector,
            (f"chunk-{chunk.index}",),
        )
        for chunk, vector in zip(chunks, await embedder.embed(leaf_texts, "document"), strict=True)
    ]
    return await build_summary_tree(current, models, embedder, cluster_size=cluster_size)


async def build_summary_tree(
    current: list[tuple],
    models: ModelGateway,
    embedder: Embedder,
    *,
    cluster_size: int = 5,
    key_prefix: str = "summary",
) -> list[SummaryNode]:
    """Build upper RAPTOR levels from any evidence-linked leaf nodes."""
    summaries: list[SummaryNode] = []
    level = 1
    while current:
        clusters = await asyncio.to_thread(_gmm_clusters, current, cluster_size)
        next_level: list[tuple[str, str, int | None, str | None, list[float]]] = []
        for cluster in clusters:
            evidence = "\n\n".join(item[1] for item in cluster)
            try:
                raw = await models.complete(
                    [
                        {"role": "system", "content": prompt("raptor-summary:v1")},
                        {"role": "user", "content": f"<evidence>\n{evidence}\n</evidence>"},
                    ],
                    json_output=True,
                )
                text = SummaryOutput.model_validate_json(raw).summary[:MAX_SUMMARY_CHARACTERS]
            except (ValueError, json.JSONDecodeError, OpenAIError) as exc:
                # One cluster's summarization failing (token-limit truncation,
                # malformed JSON) must not fail the whole corpus rebuild --
                # fall back to an extractive excerpt for just this node.
                logger.warning("summary_call_failed", extra={"reason": type(exc).__name__})
                text = evidence[:MAX_SUMMARY_CHARACTERS]
            if not is_grounded(text, evidence):
                # A summary sharing no vocabulary with its own source cluster is more
                # likely a prompt injection than a faithful abstraction; fall back to
                # an extractive excerpt rather than trust it into the navigation tree.
                logger.warning("ungrounded_summary_rejected", extra={"summary": text[:200]})
                text = evidence[:MAX_SUMMARY_CHARACTERS]
            child_ids = tuple(item[0] for item in cluster)
            # Keyed by membership, not GMM component index, so a cluster whose
            # membership is unchanged across a rebuild keeps the same node key
            # even though BIC-selected component numbering is not stable.
            key = f"{key_prefix}-{level}-{_cluster_key(child_ids)}"
            vector = (await embedder.embed([text], "document"))[0]
            node = SummaryNode(
                key=key,
                text=text,
                level=level,
                child_ids=child_ids,
                page=cluster[0][2],
                section=cluster[0][3],
                vector=vector,
                source_ids=tuple(dict.fromkeys(source for item in cluster for source in item[5])),
            )
            summaries.append(node)
            next_level.append((key, text, node.page, node.section, vector, node.source_ids))
        if len(next_level) == 1:
            break
        current = next_level
        level += 1
    if summaries:
        root = summaries[-1]
        summaries[-1] = replace(root, is_root=True)
    return summaries


def _cluster_key(child_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha1(",".join(sorted(child_ids)).encode(), usedforsecurity=False)
    return digest.hexdigest()[:12]


def _gmm_clusters(
    items: list[tuple], target_cluster_size: int, *, membership_threshold: float = 0.1
) -> list[list[tuple]]:
    """Use BIC-selected GMMs and soft membership as described by RAPTOR."""
    if len(items) <= target_cluster_size:
        return [items]
    vectors = _reduce_dimensions([item[4] for item in items])
    max_components = min(len(items), max(2, math.ceil(len(items) / target_cluster_size)))
    # A capacity-constrained first split guarantees a hierarchy; BIC still chooses
    # among valid GMM component counts and soft membership remains unchanged.
    candidates = [
        GaussianMixture(
            n_components=count,
            covariance_type="full",
            random_state=0,
            reg_covar=1e-6,
        ).fit(vectors)
        for count in range(2, max_components + 1)
    ]
    model = min(candidates, key=lambda candidate: candidate.bic(vectors))
    probabilities = model.predict_proba(vectors)
    clusters = [
        [
            item
            for item, probability in zip(items, probabilities[:, component], strict=True)
            if probability >= membership_threshold
        ]
        for component in range(model.n_components)
    ]
    return [cluster for cluster in clusters if cluster]


def _reduce_dimensions(vectors: list[list[float]]) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    components = min(32, len(values) - 1, values.shape[1])
    if components < 1 or components == values.shape[1]:
        return values
    return PCA(n_components=components, random_state=0).fit_transform(values)
