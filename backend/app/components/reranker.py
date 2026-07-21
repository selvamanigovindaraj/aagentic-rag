import re

from ..schemas.domain import Evidence


def rerank(query: str, evidence: list[Evidence], limit: int) -> list[Evidence]:
    """Cheap lexical gate before a model-backed reranker is justified by evaluation."""
    terms = {term for term in re.findall(r"[a-z0-9-]+", query.lower()) if len(term) > 2}
    relevant = [
        item for item in evidence if terms & set(re.findall(r"[a-z0-9-]+", item.text.lower()))
    ]
    return sorted(relevant, key=lambda item: item.score, reverse=True)[:limit]
