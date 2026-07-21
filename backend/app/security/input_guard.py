from ..core.errors import AppError


def validate_query(query: str, max_length: int = 20_000) -> str:
    normalized = " ".join(query.split())
    if not normalized:
        raise AppError(422, "EMPTY_QUERY", "Question cannot be empty")
    if len(normalized) > max_length:
        raise AppError(422, "QUERY_TOO_LONG", "Question exceeds the allowed length")
    return normalized
