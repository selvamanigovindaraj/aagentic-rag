CREATE TABLE IF NOT EXISTS documents (
  id uuid PRIMARY KEY, tenant_id text NOT NULL, title text NOT NULL,
  source_url text, object_key text, author text, department text,
  acl_groups text[] NOT NULL DEFAULT '{}', content_hash text, version integer NOT NULL DEFAULT 1,
  created_at timestamptz NOT NULL DEFAULT now(), deleted_at timestamptz
);
CREATE INDEX IF NOT EXISTS documents_tenant_acl_idx ON documents (tenant_id) INCLUDE (acl_groups);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
  id uuid PRIMARY KEY, tenant_id text NOT NULL, document_id uuid NOT NULL REFERENCES documents(id),
  status text NOT NULL, stage text NOT NULL, progress integer NOT NULL DEFAULT 0,
  error text, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chat_sessions (
  id uuid PRIMARY KEY, tenant_id text NOT NULL, user_id text NOT NULL, title text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chat_runs (
  id uuid PRIMARY KEY, tenant_id text NOT NULL, session_id uuid NOT NULL REFERENCES chat_sessions(id),
  user_id text NOT NULL, query text NOT NULL, status text NOT NULL, route text,
  answer text, citation_ids uuid[] NOT NULL DEFAULT '{}', error text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS citations (
  id uuid PRIMARY KEY, tenant_id text NOT NULL, document_id uuid NOT NULL,
  document_title text NOT NULL, excerpt text NOT NULL, page integer, section text
);
CREATE TABLE IF NOT EXISTS run_feedback (
  run_id uuid PRIMARY KEY REFERENCES chat_runs(id), tenant_id text NOT NULL,
  grounded boolean NOT NULL, useful boolean NOT NULL, comment text
);

