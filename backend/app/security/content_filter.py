from ..schemas.domain import Evidence


def authorized_evidence(
    evidence: list[Evidence], tenant_id: str, groups: frozenset[str]
) -> list[Evidence]:
    # Datastores filter first; this is the final fail-closed boundary before generation.
    return [
        item
        for item in evidence
        if item.source_kind == "source" and (not item.acl_groups or item.acl_groups & groups)
    ]
