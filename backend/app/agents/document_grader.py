from ..components.reranker import rerank
from ..schemas.domain import Evidence


def grade(query: str, evidence: list[Evidence], limit: int) -> list[Evidence]:
    return rerank(query, evidence, limit)
