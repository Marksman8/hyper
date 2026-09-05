"""SQLite-backed persistence.

Why raw sqlite3 and not an ORM: AURA's v0.1 data access is a dozen queries, the
schema is the interface, and an ORM would add a dependency plus a migration
framework to buy nothing at this size. If the query surface grows past ~40
statements, revisit.

Why a thread executor and not aiosqlite: sqlite3 is blocking, but AURA's write
volume is a handful of rows per conversational turn. Running them on the
default executor keeps the event loop clean without another dependency, and a
single connection guarded by a lock avoids SQLite's cross-thread rules
entirely.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aura.core.types import Fact, Permanence, Role, Session, Turn
from aura.errors import NotFoundError, StorageError
from aura.logging import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 2
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing database up to SCHEMA_VERSION.

    `CREATE TABLE IF NOT EXISTS` silently leaves an older table alone, so a
    changed CHECK constraint needs an explicit rebuild. SQLite cannot alter a
    constraint in place; copying through a new table is the supported way.
    """
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    current = int(row["v"] or 0)
    if current >= SCHEMA_VERSION:
        return

    if current < 2:
        # v2 widened audit.decision to include 'confirm'.
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='audit'"
        ).fetchone()
        if sql and "'confirm'" not in (sql["sql"] or ""):
            log.info("memory.migrating", from_version=current, to_version=2)
            conn.executescript(
                """
                PRAGMA foreign_keys = OFF;
                BEGIN;
                CREATE TABLE audit_v2 (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts               REAL NOT NULL,
                    actor            TEXT NOT NULL,
                    action           TEXT NOT NULL,
                    permission_level INTEGER NOT NULL DEFAULT 0,
                    decision         TEXT NOT NULL DEFAULT 'allow'
                                     CHECK (decision IN ('allow','confirm','deny','error')),
                    target           TEXT,
                    request_id       TEXT,
                    session_id       TEXT,
                    detail_json      TEXT NOT NULL DEFAULT '{}'
                );
                INSERT INTO audit_v2
                    SELECT id, ts, actor, action, permission_level, decision,
                           target, request_id, session_id, detail_json
                    FROM audit;
                DROP TABLE audit;
                ALTER TABLE audit_v2 RENAME TO audit;
                CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_request ON audit(request_id);
                COMMIT;
                PRAGMA foreign_keys = ON;
                """
            )


def _loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class MemoryStore:
    """Durable conversation, fact and audit storage."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Open the database and apply the schema. Synchronous, called at boot."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self._path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit BEGIN where needed
        )
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        _migrate(conn)
        conn.execute(
            "INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, time.time()),
        )
        self._conn = conn
        log.info("memory.connected", path=str(self._path), schema_version=SCHEMA_VERSION)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StorageError("MemoryStore.connect() was never called.")
        return self._conn

    async def _run(self, fn: Any, *args: Any) -> Any:
        """Execute a blocking DB callable off the event loop, serialised."""
        async with self._lock:
            return await asyncio.to_thread(fn, *args)

    # -- sessions ------------------------------------------------------------

    async def create_session(self, title: str | None = None) -> Session:
        session = Session(title=title)

        def _write() -> None:
            self._db.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at, metadata_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session.id,
                    session.title,
                    session.created_at,
                    session.updated_at,
                    json.dumps(session.metadata),
                ),
            )

        await self._run(_write)
        log.info("memory.session_created", session_id=session.id)
        return session

    async def get_session(self, session_id: str) -> Session:
        def _read() -> sqlite3.Row | None:
            cur = self._db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row: sqlite3.Row | None = cur.fetchone()
            return row

        row = await self._run(_read)
        if row is None:
            raise NotFoundError("No session with id " + repr(session_id))
        return Session(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_loads(row["metadata_json"]),
        )

    async def ensure_session(self, session_id: str | None) -> Session:
        """Fetch the named session, or create a fresh one when none is given."""
        if session_id is None:
            return await self.create_session()
        return await self.get_session(session_id)

    async def list_sessions(self, limit: int = 50) -> list[Session]:
        def _read() -> list[sqlite3.Row]:
            cur = self._db.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            )
            return cur.fetchall()

        rows = await self._run(_read)
        return [
            Session(
                id=r["id"],
                title=r["title"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                metadata=_loads(r["metadata_json"]),
            )
            for r in rows
        ]

    async def delete_session(self, session_id: str) -> bool:
        def _write() -> int:
            cur = self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount

        deleted = await self._run(_write)
        log.info("memory.session_deleted", session_id=session_id, deleted=bool(deleted))
        return bool(deleted)

    # -- turns ---------------------------------------------------------------

    async def add_turn(self, turn: Turn) -> Turn:
        def _write() -> None:
            db = self._db
            db.execute("BEGIN")
            try:
                db.execute(
                    "INSERT INTO turns(id, session_id, role, content, created_at, model, "
                    "provider, prompt_tokens, completion_tokens, latency_ms, "
                    "finish_reason, error, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        turn.id,
                        turn.session_id,
                        turn.role.value,
                        turn.content,
                        turn.created_at,
                        turn.model,
                        turn.provider,
                        turn.prompt_tokens,
                        turn.completion_tokens,
                        turn.latency_ms,
                        turn.finish_reason,
                        turn.error,
                        json.dumps(turn.metadata),
                    ),
                )
                db.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (turn.created_at, turn.session_id),
                )
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise

        await self._run(_write)
        return turn

    async def recent_turns(self, session_id: str, limit: int) -> list[Turn]:
        """Return the last `limit` turns, chronological (oldest first)."""
        if limit <= 0:
            return []

        def _read() -> list[sqlite3.Row]:
            cur = self._db.execute(
                # rowid is aliased inside the subquery because a subquery's
                # result set has no implicit rowid to order by outside it.
                "SELECT * FROM ("
                "  SELECT *, rowid AS _rid FROM turns WHERE session_id = ? "
                "  ORDER BY created_at DESC, _rid DESC LIMIT ?"
                ") ORDER BY created_at ASC, _rid ASC",
                (session_id, limit),
            )
            return cur.fetchall()

        rows = await self._run(_read)
        return [
            Turn(
                id=r["id"],
                session_id=r["session_id"],
                role=Role(r["role"]),
                content=r["content"],
                created_at=r["created_at"],
                model=r["model"],
                provider=r["provider"],
                prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                latency_ms=r["latency_ms"],
                finish_reason=r["finish_reason"],
                error=r["error"],
                metadata=_loads(r["metadata_json"]),
            )
            for r in rows
        ]

    # -- facts ---------------------------------------------------------------

    async def upsert_fact(self, fact: Fact) -> Fact:
        """Insert a fact, or reinforce an existing one.

        Reinforcement is the section 15 mechanism: repeating a preference raises
        evidence_count and confidence rather than duplicating the row.
        """

        def _write() -> None:
            self._db.execute(
                "INSERT INTO facts(id, scope, key, value, confidence, evidence_count, "
                "source, permanence, created_at, last_confirmed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(scope, key) DO UPDATE SET "
                "  value = excluded.value, "
                "  confidence = MIN(1.0, facts.confidence + 0.05), "
                "  evidence_count = facts.evidence_count + 1, "
                "  permanence = excluded.permanence, "
                "  last_confirmed_at = excluded.last_confirmed_at",
                (
                    fact.id,
                    fact.scope,
                    fact.key,
                    fact.value,
                    fact.confidence,
                    fact.evidence_count,
                    fact.source,
                    fact.permanence.value,
                    fact.created_at,
                    fact.last_confirmed_at,
                ),
            )

        await self._run(_write)
        log.info("memory.fact_upserted", scope=fact.scope, key=fact.key)
        return fact

    async def list_facts(self, scope: str | None = None) -> list[Fact]:
        def _read() -> list[sqlite3.Row]:
            if scope is None:
                cur = self._db.execute("SELECT * FROM facts ORDER BY confidence DESC, key ASC")
            else:
                cur = self._db.execute(
                    "SELECT * FROM facts WHERE scope = ? ORDER BY confidence DESC, key ASC",
                    (scope,),
                )
            return cur.fetchall()

        rows = await self._run(_read)
        return [
            Fact(
                id=r["id"],
                scope=r["scope"],
                key=r["key"],
                value=r["value"],
                confidence=r["confidence"],
                evidence_count=r["evidence_count"],
                source=r["source"],
                permanence=Permanence(r["permanence"]),
                created_at=r["created_at"],
                last_confirmed_at=r["last_confirmed_at"],
            )
            for r in rows
        ]

    async def delete_fact(self, scope: str, key: str) -> bool:
        def _write() -> int:
            cur = self._db.execute("DELETE FROM facts WHERE scope = ? AND key = ?", (scope, key))
            return cur.rowcount

        deleted = await self._run(_write)
        log.info("memory.fact_deleted", scope=scope, key=key, deleted=bool(deleted))
        return bool(deleted)

    async def forget_expired(self, now: float | None = None) -> int:
        """Drop facts whose permanence has run out (spec section 16).

        `temporary` facts live 24 hours; `session` facts are never persisted
        past the process that made them and are cleared at boot.
        """
        cutoff = (now or time.time()) - 24 * 3600

        def _write() -> int:
            cur = self._db.execute(
                "DELETE FROM facts WHERE permanence = 'session' "
                "   OR (permanence = 'temporary' AND last_confirmed_at < ?)",
                (cutoff,),
            )
            return cur.rowcount

        n: int = await self._run(_write)
        if n:
            log.info("memory.facts_expired", count=n)
        return n

    # -- audit ---------------------------------------------------------------

    async def append_audit(
        self,
        actor: str,
        action: str,
        *,
        permission_level: int = 0,
        decision: str = "allow",
        target: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        ts = time.time()

        def _write() -> None:
            self._db.execute(
                "INSERT INTO audit(ts, actor, action, permission_level, decision, "
                "target, request_id, session_id, detail_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    actor,
                    action,
                    permission_level,
                    decision,
                    target,
                    request_id,
                    session_id,
                    json.dumps(detail or {}),
                ),
            )

        await self._run(_write)

    async def recent_audit(self, limit: int = 100) -> list[dict[str, Any]]:
        def _read() -> list[sqlite3.Row]:
            cur = self._db.execute("SELECT * FROM audit ORDER BY ts DESC LIMIT ?", (limit,))
            return cur.fetchall()

        rows = await self._run(_read)
        return [
            {
                "id": r["id"],
                "ts": r["ts"],
                "actor": r["actor"],
                "action": r["action"],
                "permission_level": r["permission_level"],
                "decision": r["decision"],
                "target": r["target"],
                "request_id": r["request_id"],
                "session_id": r["session_id"],
                "detail": _loads(r["detail_json"]),
            }
            for r in rows
        ]

    # -- introspection -------------------------------------------------------

    async def stats(self) -> dict[str, int]:
        def _read() -> dict[str, int]:
            out: dict[str, int] = {}
            for table in ("sessions", "turns", "facts", "audit"):
                cur = self._db.execute("SELECT COUNT(*) AS n FROM " + table)  # noqa: S608
                out[table] = int(cur.fetchone()["n"])
            return out

        result: dict[str, int] = await self._run(_read)
        return result


def build_history(turns: Sequence[Turn], char_budget: int) -> list[Turn]:
    """Trim history to a character budget, keeping the most recent turns.

    A crude proxy for token budgeting, and deliberately so: importing a
    tokenizer to count prompt length would drag the transformers dependency
    into v0.1 to save a few hundred tokens of slack. Revisit in Phase 2, where
    a tokenizer is already present for other reasons.
    """
    kept: list[Turn] = []
    used = 0
    for turn in reversed(turns):
        cost = len(turn.content) + 16  # +16 approximates role/formatting overhead
        if used + cost > char_budget and kept:
            break
        kept.append(turn)
        used += cost
    kept.reverse()
    return kept
