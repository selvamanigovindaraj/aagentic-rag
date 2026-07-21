ALTER TABLE documents ADD COLUMN IF NOT EXISTS logical_id uuid;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS revision_of uuid;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS index_status text NOT NULL DEFAULT 'active';
UPDATE documents SET logical_id=id WHERE logical_id IS NULL;
ALTER TABLE documents ALTER COLUMN logical_id SET NOT NULL;
CREATE INDEX IF NOT EXISTS documents_tenant_hash_idx
  ON documents (tenant_id, content_hash) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS documents_logical_version_idx
  ON documents (tenant_id, logical_id, version DESC);

ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS attempts integer NOT NULL DEFAULT 0;
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS worker_id text;
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS lease_until timestamptz;
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS operation text NOT NULL DEFAULT 'index';
CREATE INDEX IF NOT EXISTS ingestion_jobs_claim_idx
  ON ingestion_jobs (status, lease_until, created_at) WHERE attempts < 3;
