from .templates import (
    EVIDENCE_CRITIC_V1,
    GROUNDED_ANSWER_V1,
    GROUNDED_CLAIMS_V1,
    LEAF_REWRITE_V1,
    OPEN_TRIPLES_V1,
    QUERY_EXPANSION_V1,
    RAPTOR_SUMMARY_V1,
    REASONING_PLAN_V1,
    ROUTER_EXPANSION_V1,
    ROUTER_V1,
)

PROMPTS = {
    "router:v1": ROUTER_V1,
    "router-expansion:v1": ROUTER_EXPANSION_V1,
    "grounded-answer:v1": GROUNDED_ANSWER_V1,
    "raptor-summary:v1": RAPTOR_SUMMARY_V1,
    "open-triples:v1": OPEN_TRIPLES_V1,
    "query-expansion:v1": QUERY_EXPANSION_V1,
    "reasoning-plan:v1": REASONING_PLAN_V1,
    "evidence-critic:v1": EVIDENCE_CRITIC_V1,
    "leaf-rewrite:v1": LEAF_REWRITE_V1,
    "grounded-claims:v1": GROUNDED_CLAIMS_V1,
}


def prompt(name: str) -> str:
    return PROMPTS[name]
