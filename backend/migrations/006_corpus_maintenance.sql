CREATE TABLE IF NOT EXISTS corpus_rebuilds (
  tenant_id text NOT NULL,
  acl_cohort text NOT NULL,
  acl_groups text[] NOT NULL,
  index_version text NOT NULL,
  status text NOT NULL DEFAULT 'queued',
  attempts integer NOT NULL DEFAULT 0,
  worker_id text,
  requested_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, acl_cohort, index_version)
);
CREATE INDEX IF NOT EXISTS corpus_rebuilds_claim_idx
  ON corpus_rebuilds (status, requested_at) WHERE attempts < 3;
