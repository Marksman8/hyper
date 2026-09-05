-- AURA schema, version 1.
--
-- Design notes:
--   * Timestamps are REAL unix epoch seconds. Cheap to compare, timezone-free,
--     and trivially convertible. ISO strings in SQLite invite subtle sort bugs.
--   * JSON lives in TEXT columns rather than separate tables. At AURA's scale
--     (one user, thousands of rows) normalising metadata buys nothing and costs
--     every future schema change.
--   * `facts` and `audit` exist in v0.1 with their full Phase 6 / Phase 3
--     column sets, unpopulated. Adding columns later is easy; back-filling
--     semantics that were never captured is not.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);

CREATE TABLE IF NOT EXISTS turns (
    id                TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role              TEXT NOT NULL CHECK (role IN ('system','user','assistant','tool')),
    content           TEXT NOT NULL,
    created_at        REAL NOT NULL,
    model             TEXT,
    provider          TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms        INTEGER NOT NULL DEFAULT 0,
    finish_reason     TEXT,
    error             TEXT,
    metadata_json     TEXT NOT NULL DEFAULT '{}'
);

-- The hot query: "last N turns of session X, oldest first".
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, created_at);

-- Preference / semantic memory (spec sections 15 and 16).
-- Nothing writes here automatically in v0.1; explicit "remember that..."
-- commands and the Phase 6 learning pipeline both land in this table.
CREATE TABLE IF NOT EXISTS facts (
    id                TEXT PRIMARY KEY,
    scope             TEXT NOT NULL DEFAULT 'user',
    key               TEXT NOT NULL,
    value             TEXT NOT NULL,
    confidence        REAL NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0.0 AND 1.0),
    evidence_count    INTEGER NOT NULL DEFAULT 1,
    source            TEXT NOT NULL DEFAULT 'explicit',
    permanence        TEXT NOT NULL DEFAULT 'long_term'
                      CHECK (permanence IN ('session','temporary','long_term','permanent')),
    created_at        REAL NOT NULL,
    last_confirmed_at REAL NOT NULL,
    UNIQUE (scope, key)
);

CREATE INDEX IF NOT EXISTS idx_facts_scope ON facts(scope, confidence DESC);

-- Append-only audit trail (spec sections 23 and 25). Never UPDATEd, never
-- DELETEd by application code.
CREATE TABLE IF NOT EXISTS audit (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               REAL NOT NULL,
    actor            TEXT NOT NULL,
    action           TEXT NOT NULL,
    permission_level INTEGER NOT NULL DEFAULT 0,
    -- 'confirm' means the action was refused for want of a human approval.
    -- It was missing from this constraint until Phase 3, which made every
    -- confirmation-gated refusal raise instead of refusing cleanly.
    decision         TEXT NOT NULL DEFAULT 'allow'
                     CHECK (decision IN ('allow','confirm','deny','error')),
    target           TEXT,
    request_id       TEXT,
    session_id       TEXT,
    detail_json      TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_request ON audit(request_id);
