"""SQLite persistence and the hash-chained ledger.

The ledger design is inherited from AGENT_BRAIN_VAULT's `brain_core.py`, which was
the most solid engineering in that system: every durable change appends an event
whose hash covers the previous event's hash, so editing or deleting history is
detectable by `verify_ledger()`. Myelin extends the event vocabulary from resource
lifecycle to cognition: model training runs, promoted configurations, accepted
links and reverted edits are all ledger events with before/after digests.

Concurrency: the daemon, the web server, and the MCP server may each hold a
connection. WAL mode plus `BEGIN IMMEDIATE` around read-last-hash/insert makes
ledger appends serial across processes. The head checkpoint is written under a
file lock and only ever moves forward, so a slow process cannot overwrite a
newer head with an older one, and a truncated ledger stays detectably truncated.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import locks

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    stem TEXT NOT NULL,
    title TEXT NOT NULL,
    type TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    frontmatter_json TEXT NOT NULL,
    frontmatter_error TEXT,
    raw_hash TEXT NOT NULL,
    mtime REAL NOT NULL,
    words INTEGER NOT NULL,
    first_seen TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_stem ON notes(stem);

CREATE TABLE IF NOT EXISTS passages (
    id INTEGER PRIMARY KEY,
    note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    heading TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_passages_note ON passages(note_id);
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
    heading, text, content='passages', content_rowid='id', tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY,
    src INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    target_raw TEXT NOT NULL,
    target_path TEXT,
    dst INTEGER REFERENCES notes(id) ON DELETE SET NULL,
    context TEXT NOT NULL,
    from_frontmatter INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_links_src ON links(src);
CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst);

CREATE TABLE IF NOT EXISTS vectors (
    owner TEXT NOT NULL,           -- 'p' passage or 'n' note
    owner_id INTEGER NOT NULL,
    encoder TEXT NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (owner, owner_id, encoder)
);

CREATE TABLE IF NOT EXISTS edges (
    src INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    dst INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,            -- explicit | related | semantic | hebbian
    weight REAL NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    updated_at REAL NOT NULL,
    PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);

CREATE TABLE IF NOT EXISTS recalls (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    query TEXT NOT NULL,
    source TEXT NOT NULL,
    session TEXT,
    config_version INTEGER,
    results_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    recall_id TEXT NOT NULL,
    note_id INTEGER NOT NULL,
    label REAL NOT NULL,           -- 1 useful, 0 not useful; implicit signals use 0.7
    source TEXT NOT NULL,
    ts REAL NOT NULL,
    UNIQUE (recall_id, note_id, source)
);

CREATE TABLE IF NOT EXISTS access (
    session TEXT NOT NULL,
    note_id INTEGER NOT NULL,
    ts REAL NOT NULL,
    consolidated INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_access_session ON access(session, consolidated);

CREATE TABLE IF NOT EXISTS proposals (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,            -- link | merge | gap | note | hub
    key TEXT UNIQUE NOT NULL,      -- dedupe identity, e.g. link:SRC->DST
    payload_json TEXT NOT NULL,
    score REAL NOT NULL,
    method TEXT NOT NULL,
    status TEXT NOT NULL,          -- pending | accepted | rejected | stale | applied | reverted
    created_at REAL NOT NULL,
    decided_at REAL,
    decided_by TEXT,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status, score);

CREATE TABLE IF NOT EXISTS gaps (
    query_norm TEXT PRIMARY KEY,
    example TEXT NOT NULL,
    count INTEGER NOT NULL,
    best_score REAL NOT NULL,
    last_seen REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    context_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name, ts);

CREATE TABLE IF NOT EXISTS configs (
    version INTEGER PRIMARY KEY,
    params_json TEXT NOT NULL,
    parent INTEGER,
    status TEXT NOT NULL,          -- champion | retired | rejected
    fitness_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    active INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (name, version)
);

CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    event_hash TEXT UNIQUE NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    mode = "wb" if isinstance(content, bytes) else "w"
    kwargs = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
    with tmp.open(mode, **kwargs) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.head_path = path.with_suffix(".head")
        path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(path.parent, 0o700)
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=30000")
        self._lock = threading.RLock()
        self._depth = 0
        self._verified: tuple[float, tuple, tuple[bool, str, int]] | None = None
        self.db.executescript(SCHEMA)
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
        self.set_meta("schema_version", str(SCHEMA_VERSION))

    def close(self) -> None:
        self.db.close()

    # ── transactions ─────────────────────────────────────────────────────
    @contextlib.contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Nested-safe IMMEDIATE transaction; the outermost commit checkpoints
        the ledger head so `verify_ledger` can detect truncation."""
        with self._lock:
            outer = self._depth == 0
            if outer:
                self.db.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self.db
            except BaseException:
                self._depth -= 1
                if outer:
                    self.db.execute("ROLLBACK")
                raise
            self._depth -= 1
            if outer:
                self.db.execute("COMMIT")
                self._write_checkpoint()

    def q(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.db.execute(sql, params))

    def one(self, sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.db.execute(sql, params).fetchone()

    def x(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.db.execute(sql, params)

    # ── meta ─────────────────────────────────────────────────────────────
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.one("SELECT value FROM meta WHERE key=?", (key,))
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.x("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
               (key, value))

    # ── ledger ───────────────────────────────────────────────────────────
    def event(self, kind: str, subject: str, payload: dict[str, Any]) -> int:
        with self.tx() as db:
            row = db.execute("SELECT event_hash FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
            previous = row[0] if row else "GENESIS"
            created = utc_now()
            body = canonical_json({"kind": kind, "subject": subject, "payload": payload, "created_at": created})
            digest = hashlib.sha256(f"{previous}\n{body}".encode()).hexdigest()
            cur = db.execute(
                "INSERT INTO events(kind,subject,payload_json,created_at,previous_hash,event_hash) VALUES(?,?,?,?,?,?)",
                (kind, subject, canonical_json(payload), created, previous, digest),
            )
            return int(cur.lastrowid)

    def _write_checkpoint(self) -> None:
        with locks.held(self.head_path.with_name(self.head_path.name + ".lock")):
            row = self.db.execute("SELECT sequence,event_hash FROM events ORDER BY sequence DESC LIMIT 1").fetchone()
            checkpoint = {"sequence": row[0] if row else 0, "event_hash": row[1] if row else "GENESIS"}
            try:
                current = json.loads(self.head_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = None
            # Forward only. An equal or later checkpoint already on disk is either
            # newer than what this process saw, or evidence of tampering; both stay.
            if current is None or int(current.get("sequence", -1)) < checkpoint["sequence"]:
                atomic_write(self.head_path, canonical_json(checkpoint) + "\n")
                with contextlib.suppress(OSError):
                    os.chmod(self.head_path, 0o600)

    def verify_ledger(self, max_age: float = 0.0) -> tuple[bool, str, int]:
        """Re-hash the whole chain. `max_age` lets a frequently polled caller
        (the status strip) reuse a result that is at most that many seconds old
        and was computed at the same head; `verify-ledger` always re-hashes."""
        head = tuple(self.one("SELECT COUNT(*), COALESCE(MAX(sequence),0) FROM events"))
        if max_age > 0 and self._verified is not None:
            at, seen_head, result = self._verified
            if seen_head == head and time.monotonic() - at < max_age:
                return result
        result = self._verify_chain()
        self._verified = (time.monotonic(), head, result)
        return result

    def _verify_chain(self) -> tuple[bool, str, int]:
        previous, last = "GENESIS", 0
        for row in self.q("SELECT * FROM events ORDER BY sequence"):
            body = canonical_json({
                "kind": row["kind"], "subject": row["subject"],
                "payload": json.loads(row["payload_json"]), "created_at": row["created_at"],
            })
            expected = hashlib.sha256(f"{previous}\n{body}".encode()).hexdigest()
            if row["previous_hash"] != previous or row["event_hash"] != expected:
                return False, f"chain mismatch at event {row['sequence']}", row["sequence"]
            previous, last = row["event_hash"], row["sequence"]
        try:
            checkpoint = json.loads(self.head_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return (last == 0), ("no events yet" if last == 0 else "checkpoint missing"), last
        if checkpoint != {"sequence": last, "event_hash": previous}:
            return False, "checkpoint does not match ledger head (truncated or rewritten)", last
        return True, f"{last} events, chain intact", last

    def events(self, limit: int = 100, after: int = 0) -> list[dict[str, Any]]:
        rows = self.q(
            "SELECT * FROM (SELECT * FROM events WHERE sequence>? ORDER BY sequence DESC LIMIT ?) ORDER BY sequence",
            (after, limit),
        )
        return [{
            "sequence": r["sequence"], "kind": r["kind"], "subject": r["subject"],
            "payload": json.loads(r["payload_json"]), "created_at": r["created_at"],
            "hash": r["event_hash"][:16],
        } for r in rows]

    # ── metrics ──────────────────────────────────────────────────────────
    def metric(self, name: str, value: float, context: dict[str, Any] | None = None) -> None:
        self.x("INSERT INTO metrics(ts,name,value,context_json) VALUES(?,?,?,?)",
               (time.time(), name, float(value), canonical_json(context or {})))

    def metric_series(self, name: str, limit: int = 200) -> list[tuple[float, float]]:
        rows = self.q("SELECT ts,value FROM (SELECT ts,value FROM metrics WHERE name=? ORDER BY ts DESC LIMIT ?) "
                      "ORDER BY ts", (name, limit))
        return [(r[0], r[1]) for r in rows]

    def latest_metric(self, name: str) -> float | None:
        row = self.one("SELECT value FROM metrics WHERE name=? ORDER BY ts DESC LIMIT 1", (name,))
        return None if row is None else float(row[0])

    # ── notes ────────────────────────────────────────────────────────────
    def note_ids(self) -> dict[str, int]:
        return {r["path"]: r["id"] for r in self.q("SELECT id,path FROM notes")}

    def note_row(self, ref: int | str) -> sqlite3.Row | None:
        if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
            return self.one("SELECT * FROM notes WHERE id=?", (int(ref),))
        row = self.one("SELECT * FROM notes WHERE path=?", (ref,))
        if row is None:
            row = self.one("SELECT * FROM notes WHERE stem=? ORDER BY length(path) LIMIT 1", (ref,))
        if row is None:
            row = self.one("SELECT * FROM notes WHERE lower(stem)=lower(?) OR lower(title)=lower(?) "
                           "ORDER BY length(path) LIMIT 1", (ref, ref))
        return row
