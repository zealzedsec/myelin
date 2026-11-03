"""Learning from use: feedback labels, implicit reads, and Hebbian co-activation.

"Neurons that fire together wire together." When notes are used in the same
working session — read by Claude Code after a recall, or marked useful together —
the edge between them strengthens by a bounded step, w ← w + η(1 − w). Unused
edges decay with a half-life and are pruned below a floor, so the learned layer
of the graph tracks how the vault is actually used rather than growing forever.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any

from .brain import Brain
from .graph import hebbian_effective, hebbian_update

IMPLICIT_LABEL = 0.7          # reading a recalled note is evidence, weaker than a 👍
SESSION_WINDOW_SECONDS = 1800
MAX_SESSION_NOTES = 12


def resolve_note(brain: Brain, ref: str | int) -> int | None:
    if isinstance(ref, int):
        row = brain.store.one("SELECT id FROM notes WHERE id=?", (ref,))
        return row[0] if row else None
    ref = str(ref)
    root = str(brain.cfg.root.resolve())
    if ref.startswith(root):
        ref = Path(ref).resolve().relative_to(brain.cfg.root.resolve()).as_posix()
    row = brain.store.note_row(ref)
    return int(row["id"]) if row else None


def record_feedback(brain: Brain, recall_id: str, note_ref: str | int, label: float, source: str) -> dict[str, Any]:
    note_id = resolve_note(brain, note_ref)
    if note_id is None:
        raise KeyError(f"unknown note {note_ref!r}")
    if brain.store.one("SELECT 1 FROM recalls WHERE id=?", (recall_id,)) is None:
        raise KeyError(f"unknown recall {recall_id!r}")
    label = float(min(max(label, 0.0), 1.0))
    brain.store.x("INSERT INTO feedback(recall_id,note_id,label,source,ts) VALUES(?,?,?,?,?) "
                  "ON CONFLICT(recall_id,note_id,source) DO UPDATE SET label=excluded.label, ts=excluded.ts",
                  (recall_id, note_id, label, source, time.time()))
    reinforced = 0
    if label >= 0.5:
        others = [r[0] for r in brain.store.q(
            "SELECT DISTINCT note_id FROM feedback WHERE recall_id=? AND label>=0.5 AND note_id!=?",
            (recall_id, note_id))]
        for other in others:
            hebbian_reinforce(brain, note_id, other, 0.5)
            reinforced += 1
    return {"recall_id": recall_id, "note_id": note_id, "label": label, "reinforced": reinforced}


def record_access(brain: Brain, session: str, note_ref: str | int) -> int | None:
    """A note was read in a working session. If it appeared in a recall made in
    the same session recently, that read is an implicit 'this was useful'."""
    note_id = resolve_note(brain, note_ref)
    if note_id is None:
        return None
    now = time.time()
    brain.store.x("INSERT INTO access(session,note_id,ts) VALUES(?,?,?)", (session, note_id, now))
    # Same session first. An MCP server cannot see Claude Code's session id, so
    # its recalls carry their own; a read shortly after an MCP recall that
    # returned this note is credited to it as the next-best evidence.
    candidates = brain.store.q(
        "SELECT id,results_json FROM recalls WHERE session=? AND ts>? ORDER BY ts DESC LIMIT 5",
        (session, now - SESSION_WINDOW_SECONDS)) or brain.store.q(
        "SELECT id,results_json FROM recalls WHERE source='mcp' AND ts>? ORDER BY ts DESC LIMIT 5",
        (now - 600,))
    for r in candidates:
        if any(item["note_id"] == note_id for item in json.loads(r["results_json"])):
            brain.store.x("INSERT OR IGNORE INTO feedback(recall_id,note_id,label,source,ts) VALUES(?,?,?,?,?)",
                          (r["id"], note_id, IMPLICIT_LABEL, "implicit-read", now))
            break
    return note_id


def hebbian_reinforce(brain: Brain, a: int, b: int, strength: float = 1.0) -> float:
    if a == b:
        return 0.0
    src, dst = sorted((a, b))
    now = time.time()
    row = brain.store.one("SELECT weight,updated_at,count FROM edges WHERE src=? AND dst=? AND kind='hebbian'",
                          (src, dst))
    current = hebbian_effective(row[0], row[1], brain.cfg.hebbian_half_life_days, now) if row else 0.0
    weight = hebbian_update(current, brain.cfg.hebbian_lr, strength)
    brain.store.x("INSERT INTO edges(src,dst,kind,weight,count,updated_at) VALUES(?,?,'hebbian',?,1,?) "
                  "ON CONFLICT(src,dst,kind) DO UPDATE SET weight=excluded.weight, count=count+1, "
                  "updated_at=excluded.updated_at", (src, dst, weight, now))
    brain._bump("graph_version")
    return weight


def consolidate_sessions(brain: Brain) -> dict[str, int]:
    sessions = [r[0] for r in brain.store.q("SELECT DISTINCT session FROM access WHERE consolidated=0")]
    pairs = 0
    for session in sessions:
        rows = brain.store.q("SELECT note_id, MAX(ts) t FROM access WHERE session=? GROUP BY note_id "
                             "ORDER BY t DESC LIMIT ?", (session, MAX_SESSION_NOTES))
        notes = [r[0] for r in rows]
        # Strength shrinks with session breadth: two notes read together say more
        # about each other than two of twelve do.
        strength = 1.0 / max(1.0, (len(notes) - 1) ** 0.5)
        for a, b in combinations(notes, 2):
            hebbian_reinforce(brain, a, b, strength)
            pairs += 1
        brain.store.x("UPDATE access SET consolidated=1 WHERE session=?", (session,))
    return {"sessions": len(sessions), "pairs": pairs}


def prune_hebbian(brain: Brain) -> int:
    now = time.time()
    doomed = [(r[0], r[1]) for r in brain.store.q("SELECT src,dst,weight,updated_at FROM edges WHERE kind='hebbian'")
              if hebbian_effective(r[2], r[3], brain.cfg.hebbian_half_life_days, now) < brain.cfg.hebbian_prune_below]
    with brain.store.tx() as db:
        db.executemany("DELETE FROM edges WHERE src=? AND dst=? AND kind='hebbian'", doomed)
    if doomed:
        brain._bump("graph_version")
    return len(doomed)
