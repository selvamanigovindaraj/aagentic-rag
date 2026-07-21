# API reference

Interactive OpenAPI documentation is served at `/docs`. All `/api/v1` routes require OIDC in production. Development mode uses `X-Tenant-ID`, `X-User-ID`, and optional comma-separated `X-Groups` headers.

Chat answers stream as server-sent events: `status`, `answer`, `complete`, or `error`. Citation identifiers resolve through `/api/v1/citations/{id}` and are tenant-scoped.
