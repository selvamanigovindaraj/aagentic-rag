ALTER TABLE chat_runs ADD COLUMN IF NOT EXISTS groups text[] NOT NULL DEFAULT '{}';
ALTER TABLE chat_runs ADD COLUMN IF NOT EXISTS attempts integer NOT NULL DEFAULT 0;
ALTER TABLE chat_runs ADD COLUMN IF NOT EXISTS worker_id text;
ALTER TABLE chat_runs ADD COLUMN IF NOT EXISTS lease_until timestamptz;
CREATE INDEX IF NOT EXISTS chat_runs_claim_idx
  ON chat_runs (status, lease_until, created_at) WHERE attempts < 3;
