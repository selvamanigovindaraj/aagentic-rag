from app.components.retrieval import EmptyRetriever
from app.core.config import Settings
from app.main import app
from app.repositories.store import MemoryStore
from app.schemas.domain import Citation, Document
from app.services.events import MemoryEventBroker
from app.services.rag_pipeline import RagPipeline
from fastapi.testclient import TestClient

HEADERS = {"X-Tenant-ID": "tenant-a", "X-User-ID": "user-a", "X-Groups": "hr"}


def configure_test_app() -> None:
    app.state.store = MemoryStore()
    app.state.events = MemoryEventBroker()
    app.state.settings = Settings()
    app.state.agent = RagPipeline(EmptyRetriever(), app.state.settings)


def test_document_and_chat_workflow():
    configure_test_app()
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/documents",
            headers=HEADERS,
            json={"title": "Policy", "object_key": "tenant-a/policy.pdf", "acl_groups": ["hr"]},
        )
        assert created.status_code == 202, created.text
        job_id = created.json()["ingestion_job"]["id"]
        assert client.get(f"/api/v1/ingestion-jobs/{job_id}", headers=HEADERS).status_code == 200
        assert len(client.get("/api/v1/documents", headers=HEADERS).json()) == 1

        session = client.post("/api/v1/chat/sessions", headers=HEADERS, json={}).json()
        run = client.post(
            f"/api/v1/chat/sessions/{session['id']}/messages",
            headers=HEADERS,
            json={"content": "What is the policy?"},
        )
        assert run.status_code == 202
        events = client.get(run.json()["events_url"], headers=HEADERS)
        assert events.status_code == 200
        assert '"type": "complete"' in events.text


def test_cross_tenant_job_is_not_disclosed():
    configure_test_app()
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/documents",
            headers=HEADERS,
            json={"title": "Secret", "object_key": "secret.pdf"},
        ).json()
        response = client.get(
            f"/api/v1/ingestion-jobs/{created['ingestion_job']['id']}",
            headers={"X-Tenant-ID": "tenant-b", "X-User-ID": "user-b"},
        )
        assert response.status_code == 404


def test_api_exports_latency_metric_for_autoscaling():
    configure_test_app()
    with TestClient(app) as client:
        client.get("/api/v1/documents", headers=HEADERS)
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.text.startswith("rag_api_request_latency_p95_seconds ")


def test_document_upload_is_persisted_and_queued(tmp_path):
    configure_test_app()
    app.state.settings = Settings(object_storage_path=str(tmp_path))
    app.state.ingestion_jobs = []
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/documents",
            headers={key: value for key, value in HEADERS.items() if key != "Content-Type"},
            data={"title": "Architecture", "acl_groups": "research"},
            files={
                "file": ("architecture.txt", b"RAPTOR builds recursive summaries.", "text/plain")
            },
        )

    assert response.status_code == 202, response.text
    accepted = response.json()
    path = tmp_path / accepted["document"]["object_key"]
    assert path.read_text() == "RAPTOR builds recursive summaries."
    assert accepted["document"]["content_hash"]
    assert app.state.ingestion_jobs == [accepted["ingestion_job"]["id"]]


def test_document_upload_accepts_files_larger_than_starlette_default(tmp_path):
    configure_test_app()
    app.state.settings = Settings(object_storage_path=str(tmp_path))
    content = b"x" * (1024 * 1024 + 1)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/documents",
            headers=HEADERS,
            files={"file": ("large.txt", content, "text/plain")},
        )

    assert response.status_code == 202, response.text
    path = tmp_path / response.json()["document"]["object_key"]
    assert path.stat().st_size == len(content)


def test_duplicate_upload_reuses_the_authorized_document_and_job(tmp_path):
    configure_test_app()
    app.state.settings = Settings(object_storage_path=str(tmp_path))
    app.state.ingestion_jobs = []
    with TestClient(app) as client:
        first = client.post(
            "/api/v1/documents",
            headers=HEADERS,
            files={"file": ("policy.txt", b"same policy", "text/plain")},
        ).json()
        second = client.post(
            "/api/v1/documents",
            headers=HEADERS,
            files={"file": ("copy.txt", b"same policy", "text/plain")},
        ).json()

    assert second["document"]["id"] == first["document"]["id"]
    assert second["ingestion_job"]["id"] == first["ingestion_job"]["id"]
    assert len(app.state.store.documents) == 1


def test_source_url_without_object_reference_is_rejected():
    configure_test_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/documents",
            headers=HEADERS,
            json={"title": "Remote", "source_url": "https://example.com/document.pdf"},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "OBJECT_REFERENCE_REQUIRED"


def test_revision_creates_an_immutable_next_version(tmp_path):
    configure_test_app()
    app.state.settings = Settings(object_storage_path=str(tmp_path))
    app.state.ingestion_jobs = []
    with TestClient(app) as client:
        first = client.post(
            "/api/v1/documents",
            headers=HEADERS,
            files={"file": ("policy.txt", b"version one", "text/plain")},
        ).json()["document"]
        second = client.post(
            "/api/v1/documents",
            headers=HEADERS,
            data={"revision_of": first["id"]},
            files={"file": ("policy.txt", b"version two", "text/plain")},
        ).json()["document"]

    assert first["version"] == 1
    assert second["version"] == 2
    assert second["logical_id"] == first["logical_id"]
    assert second["id"] != first["id"]


def test_invalid_revision_identifier_is_rejected(tmp_path):
    configure_test_app()
    app.state.settings = Settings(object_storage_path=str(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/documents",
            headers=HEADERS,
            data={"revision_of": "not-a-uuid"},
            files={"file": ("policy.txt", b"version", "text/plain")},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REVISION"


def test_delete_fails_closed_and_queues_index_cleanup(tmp_path):
    configure_test_app()
    app.state.settings = Settings(object_storage_path=str(tmp_path))
    with TestClient(app) as client:
        document = client.post(
            "/api/v1/documents",
            headers=HEADERS,
            files={"file": ("policy.txt", b"policy", "text/plain")},
        ).json()["document"]
        response = client.delete(f"/api/v1/documents/{document['id']}", headers=HEADERS)
        visible_documents = client.get("/api/v1/documents", headers=HEADERS).json()

    assert response.status_code == 204
    assert visible_documents == []
    cleanup = [job for job in app.state.store.jobs.values() if job.operation == "delete"]
    assert len(cleanup) == 1 and str(cleanup[0].document_id) == document["id"]


def test_delete_requires_uploader_not_just_acl_membership():
    configure_test_app()
    app.state.settings = Settings()
    other_user = {"X-Tenant-ID": "tenant-a", "X-User-ID": "user-b", "X-Groups": "hr"}
    with TestClient(app) as client:
        document = client.post(
            "/api/v1/documents",
            headers=HEADERS,
            json={"title": "Policy", "object_key": "tenant-a/policy.pdf", "acl_groups": ["hr"]},
        ).json()["document"]
        rejected = client.delete(f"/api/v1/documents/{document['id']}", headers=other_user)
        still_visible = client.get("/api/v1/documents", headers=HEADERS).json()
        accepted = client.delete(f"/api/v1/documents/{document['id']}", headers=HEADERS)

    assert rejected.status_code == 404
    assert still_visible[0]["id"] == document["id"]
    assert accepted.status_code == 204


def test_citation_requires_document_acl():
    configure_test_app()
    document = Document(tenant_id="tenant-a", title="Private", acl_groups={"legal"})
    citation = Citation(
        tenant_id="tenant-a",
        document_id=document.id,
        document_title=document.title,
        excerpt="private evidence",
    )
    app.state.store.documents[document.id] = document
    app.state.store.citations[citation.id] = citation
    with TestClient(app) as client:
        response = client.get(f"/api/v1/citations/{citation.id}", headers=HEADERS)
    assert response.status_code == 404


def test_feedback_requires_run_owner():
    configure_test_app()
    with TestClient(app) as client:
        session = client.post("/api/v1/chat/sessions", headers=HEADERS, json={}).json()
        run = client.post(
            f"/api/v1/chat/sessions/{session['id']}/messages",
            headers=HEADERS,
            json={"content": "What is the policy?"},
        ).json()
        response = client.post(
            f"/api/v1/chat/runs/{run['run_id']}/feedback",
            headers={**HEADERS, "X-User-ID": "user-b"},
            json={"grounded": True, "useful": True},
        )

    assert response.status_code == 404
    assert app.state.store.feedback == {}
