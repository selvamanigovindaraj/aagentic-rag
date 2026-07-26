ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS index_version text NOT NULL DEFAULT 'v1';
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS model_calls integer NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS ingestion_job_events (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  job_id uuid NOT NULL REFERENCES ingestion_jobs(id) ON DELETE CASCADE,
  tenant_id text NOT NULL,
  status text NOT NULL,
  stage text NOT NULL,
  progress integer NOT NULL,
  attempt integer NOT NULL,
  model_calls integer NOT NULL,
  index_version text NOT NULL,
  error text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ingestion_job_events_job_created_idx
  ON ingestion_job_events (job_id, created_at);
