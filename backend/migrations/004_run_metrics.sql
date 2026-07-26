ALTER TABLE chat_runs ADD COLUMN IF NOT EXISTS metrics jsonb NOT NULL DEFAULT '{}';
CREATE INDEX IF NOT EXISTS chat_runs_route_created_idx ON chat_runs (route, created_at DESC);
