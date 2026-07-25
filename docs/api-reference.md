# API reference

Interactive OpenAPI documentation is served at `/docs`. All `/api/v1` routes require OIDC in
production; development mode (`DEV_AUTH=true`) uses `X-Tenant-ID`, `X-User-ID`, and optional
comma-separated `X-Groups` headers instead. Every request is additionally rate-limited per
tenant+subject to `requests_per_minute` (120/min by default) via a Redis counter; requests over
the limit get `429 RATE_LIMITED`. `/healthz` and `/metrics` (Prometheus-scrapeable request p95
latency) sit outside `/api/v1` and need no auth.

Errors are a consistent JSON shape from every endpoint: `{"error": {"code": "...", "message":
"..."}}`, with the HTTP status carrying the same meaning as `code` (e.g. `404
DOCUMENT_NOT_FOUND`, `422 EMPTY_DOCUMENT`, `429 RATE_LIMITED`).

## Documents

- `POST /api/v1/documents` — upload (`multipart/form-data`: `file`, `title`, `acl_groups`,
  optional `revision_of`) or register by reference (JSON `DocumentCreate` body with an
  `object_key` already in storage). Returns `202` with `DocumentAccepted` (the document plus its
  queued `IngestionJob`); re-uploading identical content (`content_hash` match) returns the
  existing document/job instead of a duplicate. `revision_of` creates an immutable next version
  and supersedes the prior one once indexing completes.
- `GET /api/v1/ingestion-jobs/{job_id}` — poll ingestion progress/status.
- `GET /api/v1/documents` — list documents visible to the caller's tenant + ACL groups.
- `DELETE /api/v1/documents/{document_id}` — `204`; queues async index cleanup (Weaviate, Neo4j,
  object storage, layout manifest) rather than deleting synchronously.

## Chat

- `POST /api/v1/chat/sessions` — `201`, creates a `ChatSession`.
- `POST /api/v1/chat/sessions/{session_id}/messages` — `202`, queues a `ChatRun` and returns
  `RunAccepted` with an `events_url` to stream progress from.
- `GET /api/v1/chat/runs/{run_id}/events` — Server-Sent Events: `status` (stage name), `token`
  (streamed answer tokens), `answer` (final content + citation IDs), `complete`, or `error`. A
  run already `complete` when polled replays its stored answer as one `answer` + `complete` pair
  instead of an empty stream.
- `POST /api/v1/chat/runs/{run_id}/feedback` — `204`, records `{"grounded": bool, "useful": bool,
  "comment": str | null}` against a run for the evaluation flywheel.
- `GET /api/v1/citations/{citation_id}` — tenant-scoped citation lookup (only resolves citations
  belonging to a currently active, authorized document).

## Notes for integrators

- `object_key` is required on direct JSON document creation; `source_url` is stored as metadata
  only and is never fetched by the server.
- ACL enforcement happens before scoring in every retrieval path, not just at these endpoints —
  a group absent from `X-Groups`/the OIDC token's `groups` claim never sees that document's
  evidence, chunks, or citations, regardless of route.
