def rewrite(query: str, attempt: int) -> str:
    return query if attempt == 1 else f"{query} relevant evidence source"
