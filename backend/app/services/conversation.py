from uuid import UUID

from ..repositories.store import Store
from ..schemas.domain import ChatSession


async def require_owned_session(
    store: Store, tenant_id: str, user_id: str, session_id: UUID
) -> ChatSession | None:
    session = await store.get_session(tenant_id, session_id)
    return session if session and session.user_id == user_id else None
