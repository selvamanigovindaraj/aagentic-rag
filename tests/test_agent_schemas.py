import pytest
from app.prompts.registry import prompt
from app.schemas.agent import GroundedAnswer, QueryExpansion, ReasoningPlan
from pydantic import ValidationError


def test_reasoning_plan_rejects_unknown_dependencies():
    with pytest.raises(ValidationError):
        ReasoningPlan.model_validate(
            {
                "known_entities": [],
                "unknown_entities": [],
                "leaves": [{"id": "leaf", "question": "Find it", "depends_on": ["missing"]}],
            }
        )


def test_retrieval_expansion_and_claims_are_strictly_structured():
    expansion = QueryExpansion.model_validate_json(
        '{"expanded_query":"Rayleigh scattering wavelength","terms":["wavelength"]}'
    )
    answer = GroundedAnswer.model_validate_json(
        '{"claims":[{"text":"The policy changed.","evidence_ids":["source-1"]}],"unsupported":[]}'
    )

    assert "retrieval-only" in prompt("query-expansion:v1")
    assert "untrusted" in prompt("reasoning-plan:v1")
    assert expansion.terms == ["wavelength"]
    assert answer.claims[0].evidence_ids == ["source-1"]
