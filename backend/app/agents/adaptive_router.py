from ..schemas.domain import Route


def classify(query: str) -> Route:
    text = query.lower()
    if any(term in text for term in ("ignore previous", "write malware")):
        return Route.OUT_OF_SCOPE
    if any(term in text for term in ("compare", "versus", " vs ", "difference between")):
        return Route.COMPARISON
    if any(term in text for term in ("cause", "timeline", "before", "after", "over time")):
        return Route.TEMPORAL
    if any(term in text for term in ("connection between", "link between", "how is", "multi-hop")):
        return Route.MULTIHOP
    if any(term in text for term in ("summarize", "overview", "trend", "across all")):
        return Route.SYNTHESIS
    return Route.DIRECT
