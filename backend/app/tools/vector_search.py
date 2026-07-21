from ..components.retrieval import Retriever
from ..schemas.domain import AuthContext, Evidence, Route


async def vector_search(
    retriever: Retriever, query: str, auth: AuthContext, limit: int
) -> list[Evidence]:
    """Search authorized hybrid vector/lexical evidence."""
    return await retriever.retrieve(query, Route.DIRECT, auth, limit)
