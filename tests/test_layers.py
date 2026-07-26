from pathlib import Path
from uuid import uuid4

from app.schemas.domain import Evidence
from app.security.content_filter import authorized_evidence
from app.security.output_filter import valid_citation_numbers

from evaluation.offline_eval import route_accuracy
from evaluation.online_monitor import QualityWindow
from evaluation.query_latency import percentile_95
from evaluation.retrieval_eval import recall_at_k
from evaluation.runtime_report import QUERY


def test_security_layers_fail_closed():
    evidence = Evidence(
        id="1",
        document_id=uuid4(),
        document_title="Private",
        text="Revenue is 42.",
        score=1,
        acl_groups={"finance"},
    )
    assert authorized_evidence([evidence], "tenant", frozenset({"finance"})) == [evidence]
    assert authorized_evidence([evidence], "tenant", frozenset({"hr"})) == []
    assert valid_citation_numbers("Revenue is 42 [1].", 1)
    assert not valid_citation_numbers("Revenue is 42 [2].", 1)


def test_golden_routes_are_stable():
    dataset = Path(__file__).parents[1] / "evaluation" / "golden_dataset.json"
    assert route_accuracy(dataset) == 1


def test_frontend_proxy_reresolves_recreated_api_container():
    config = (Path(__file__).parents[1] / "frontend" / "nginx.conf").read_text()

    assert "resolver 127.0.0.11" in config
    assert "proxy_pass $api_upstream" in config
    assert "client_max_body_size 64m" in config


def test_quality_metrics_are_weighted_at_claim_and_citation_level():
    window = QualityWindow(
        grounded_claims=95,
        generated_claims=100,
        valid_citation_references=99,
        citation_references=100,
    )

    assert window.grounded_claim_rate == 0.95
    assert window.citation_validity == 0.99
    assert "AS grounded_claim_rate" in QUERY
    assert "AS citation_validity" in QUERY


def test_retrieval_recall_counts_expected_documents_once():
    retrieved = ["Policy A", "Policy A", "Policy B", "Noise"]

    assert recall_at_k(retrieved, {"Policy A", "Policy B", "Policy C"}, 20) == 2 / 3
    assert recall_at_k(retrieved, set(), 20) == 1.0


def test_latency_gate_uses_nearest_rank_p95():
    assert percentile_95([100, 200, 300, 400, 500]) == 500
    assert percentile_95([42]) == 42
