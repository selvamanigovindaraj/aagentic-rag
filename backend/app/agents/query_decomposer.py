import re

from ..schemas.domain import Route


def decompose(query: str, route: Route) -> list[str]:
    if route == Route.COMPARISON:
        parts = re.split(r"\b(?:versus|vs\.?|and)\b", query, maxsplit=1, flags=re.IGNORECASE)
        return [part.strip(" ?") for part in parts if part.strip(" ?")]
    return [query]
