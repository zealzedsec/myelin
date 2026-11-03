"""The proposal inbox: every change Myelin wants to make to the vault, staged.

Myelin never edits a note on its own judgement. It *proposes*; a person (web UI,
CLI) or an explicitly authorised agent accepts or rejects. This mirrors the
vault's own invariant — "no direct memory writes; adapters stage artifacts" —
and every decision is training data: an accepted link is a positive label for
the link predictor, a rejected one is a hard negative that is never proposed
again.

Applying a link is a narrow, reversible edit: one bullet inside a delimited
`## Associations` block at the end of the source note, written atomically, with
the before/after SHA-256 recorded in the ledger. `revert` removes exactly that
bullet, and refuses if the file changed in the meantime.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from .brain import Brain
from .store import atomic_write, canonical_json

START = "<!-- myelin:associations -->"
END = "<!-- /myelin:associations -->"
SECTION = "## Associations"


def is_protected(cfg, rel: str) -> bool:
    return any(fnmatch.fnmatchcase(rel, pat) or fnmatch.fnmatchcase("/" + rel, pat) for pat in cfg.protected)


def upsert(brain: Brain, kind: str, key: str, payload: dict[str, Any], score: float, method: str) -> bool:
    """Insert a proposal unless one with the same key exists in any state —
    a rejected proposal must stay rejected, not come back next cycle."""
    cur = brain.store.x(
        "INSERT OR IGNORE INTO proposals(kind,key,payload_json,score,method,status,created_at) "
        "VALUES(?,?,?,?,?,'pending',?)", (kind, key, canonical_json(payload), float(score), method, time.time()))
    if cur.rowcount:
        return True
    brain.store.x("UPDATE proposals SET score=?, payload_json=?, method=? WHERE key=? AND status='pending'",
                  (float(score), canonical_json(payload), method, key))
    return False


def link_key(a: str, b: str) -> str:
    return "link:" + "|".join(sorted((a, b)))


def listing(brain: Brain, status: str = "pending", kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    sql = "SELECT * FROM proposals WHERE status=?"
    args: list[Any] = [status]
    if kind:
        sql += " AND kind=?"
        args.append(kind)
    sql += " ORDER BY COALESCE(json_extract(payload_json,'$.utility'), score) DESC, id LIMIT ?"
    args.append(limit)
    return [_row(r) for r in brain.store.q(sql, args)]


def _row(r) -> dict[str, Any]:
    return {"id": r["id"], "kind": r["kind"], "key": r["key"], "payload": json.loads(r["payload_json"]),
            "score": round(r["score"], 4), "method": r["method"], "status": r["status"],
            "created_at": r["created_at"], "decided_at": r["decided_at"], "decided_by": r["decided_by"],
            "reason": r["reason"]}


def get(brain: Brain, pid: int) -> dict[str, Any] | None:
    r = brain.store.one("SELECT * FROM proposals WHERE id=?", (pid,))
    return _row(r) if r else None


def decide(brain: Brain, pid: int, accept: bool, actor: str, reason: str = "") -> dict[str, Any]:
    prop = get(brain, pid)
    if prop is None:
        raise KeyError(f"no proposal {pid}")
    # Claim it in one statement, so a double click, or a person and an agent
    # deciding at the same moment, applies the edit once. The loser is told.
    claimed = brain.store.x("UPDATE proposals SET status='deciding' WHERE id=? AND status='pending'", (pid,)).rowcount
    if not claimed:
        current = get(brain, pid)
        raise ValueError(f"proposal {pid} is already {current['status'] if current else 'gone'}")
    outcome: dict[str, Any] = {"id": pid, "accepted": accept}
    status = "rejected"
    try:
        if accept:
            status = "accepted"
            if prop["kind"] == "link":
                outcome.update(apply_link(brain, prop))
                status = "applied"
            elif prop["kind"] == "note":
                outcome.update(apply_note(brain, prop))
                status = "applied"
    except FileNotFoundError:
        brain.store.x("UPDATE proposals SET status='stale' WHERE id=?", (pid,))
        raise
    except BaseException:
        brain.store.x("UPDATE proposals SET status='pending' WHERE id=?", (pid,))
        raise
    brain.store.x("UPDATE proposals SET status=?, decided_at=?, decided_by=?, reason=? WHERE id=?",
                  (status, time.time(), actor, reason[:500], pid))
    recorded = {k: v for k, v in outcome.items() if k.endswith("sha256") or k == "changed"}
    brain.store.event("PROPOSAL_" + ("ACCEPTED" if accept else "REJECTED"), prop["key"], {
        "id": pid, "kind": prop["kind"], "score": prop["score"], "method": prop["method"],
        "actor": actor, "reason": reason[:200], **recorded,
    })
    outcome["status"] = status
    return outcome


def _block(text: str) -> tuple[int, int] | None:
    """Where the managed block's body starts and ends. Anchored on the first
    END and the last START before it, so a stray marker left by hand elsewhere
    in the note can never make the block swallow the author's own text."""
    end = text.find(END)
    if end == -1:
        return None
    start = text.rfind(START, 0, end)
    if start == -1:
        return None
    return start + len(START), end


def _note_target(brain: Brain, path: str) -> str:
    """The shortest wikilink text that resolves uniquely to `path`."""
    stem = Path(path).stem
    dupes = json.loads(brain.store.get_meta("duplicate_stems", "{}") or "{}")
    return path[:-3] if stem in dupes else stem


def apply_link(brain: Brain, prop: dict[str, Any]) -> dict[str, Any]:
    payload = prop["payload"]
    src_rel, dst_rel = payload["src_path"], payload["dst_path"]
    if is_protected(brain.cfg, src_rel):
        raise PermissionError(f"{src_rel} is protected (regenerated by another program)")
    path = _vault_file(brain, src_rel)
    if not path.is_file():
        raise FileNotFoundError(src_rel)
    before = path.read_text(encoding="utf-8")
    target = _note_target(brain, dst_rel)
    # `why` can come from an agent. Strip anything that would become structure
    # in the note: extra links, or markers that could close the managed block.
    reason = re.sub(r"\[\[|\]\]|<!--|-->", "", payload.get("why", "")).replace("\n", " ").strip()[:160]
    bullet = f"- [[{target}]]" + (f" — {reason}" if reason else "")
    after = add_association(before, target, bullet)
    if after == before:
        return {"changed": False}
    atomic_write(path, after)
    return {"changed": True, "path": src_rel, "line": bullet,
            "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(after.encode()).hexdigest()}


def _vault_file(brain: Brain, rel: str) -> Path:
    path = (brain.cfg.root / rel).resolve()
    if brain.cfg.root.resolve() not in path.parents:
        raise PermissionError("path escapes the vault root")
    return path


def add_association(text: str, target: str, bullet: str) -> str:
    span = _block(text)
    if span:
        a, b = span
        block = text[a:b]
        if f"[[{target}]]" in block:
            return text
        lines = [ln for ln in block.strip("\n").split("\n") if ln.strip()]
        lines.append(bullet)
        return text[:a] + "\n" + "\n".join(lines) + "\n" + text[b:]
    sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return f"{text}{sep}\n{SECTION}\n\n{START}\n{bullet}\n{END}\n"


def revert(brain: Brain, pid: int, actor: str) -> dict[str, Any]:
    prop = get(brain, pid)
    if prop is None or prop["status"] != "applied" or prop["kind"] != "link":
        raise ValueError("only applied link proposals can be reverted")
    event = brain.store.one("SELECT payload_json FROM events WHERE kind='PROPOSAL_ACCEPTED' AND subject=? "
                            "ORDER BY sequence DESC LIMIT 1", (prop["key"],))
    record = json.loads(event[0]) if event else {}
    if record.get("changed") is False:
        # The link was already in the note when this was accepted, so the line
        # is the author's own. Reverting the decision must not delete it.
        brain.store.x("UPDATE proposals SET status='reverted', decided_by=? WHERE id=?", (actor, pid))
        brain.store.event("PROPOSAL_REVERTED", prop["key"], {"id": pid, "actor": actor, "changed": False})
        return {"reverted": True, "changed": False}
    after_sha = record.get("after_sha256")
    if not after_sha:
        raise RuntimeError("no record of the edit this proposal made; revert by hand")
    path = _vault_file(brain, prop["payload"]["src_path"])
    current = path.read_text(encoding="utf-8")
    if hashlib.sha256(current.encode()).hexdigest() != after_sha:
        raise RuntimeError("the note changed after the link was applied; revert by hand")
    target = _note_target(brain, prop["payload"]["dst_path"])
    span = _block(current)
    if span is None:
        raise RuntimeError("the associations block is gone; revert by hand")
    a, b = span
    head, block, tail = current[:a - len(START)], current[a:b], current[b + len(END):]
    lines = [ln for ln in block.strip("\n").split("\n") if ln.strip() and f"[[{target}]]" not in ln]
    if lines:
        updated = head + START + "\n" + "\n".join(lines) + "\n" + END + tail
    else:
        updated = head.rstrip().removesuffix(SECTION).rstrip() + "\n" + tail.lstrip("\n")
    atomic_write(path, updated)
    brain.store.x("UPDATE proposals SET status='reverted', decided_by=? WHERE id=?", (actor, pid))
    brain.store.event("PROPOSAL_REVERTED", prop["key"], {"id": pid, "actor": actor})
    return {"reverted": True}


def apply_note(brain: Brain, prop: dict[str, Any]) -> dict[str, Any]:
    """Write an agent-drafted note into the configured inbox folder. Never
    overwrites: an existing file with that name is an error."""
    payload = prop["payload"]
    name = "".join(ch for ch in payload["title"] if ch.isalnum() or ch in " _-").strip().replace(" ", "_")[:80]
    if not name:
        raise ValueError("note has no usable title")
    inbox = brain.cfg.root / brain.cfg.inbox_dir
    path = inbox / f"{name}.md"
    if path.exists():
        raise FileExistsError(path.name)
    content = payload["content"]
    atomic_write(path, content)
    return {"path": path.relative_to(brain.cfg.root).as_posix(),
            "after_sha256": hashlib.sha256(content.encode()).hexdigest()}


def link_labels(brain: Brain) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Human decisions on link proposals as (accepted, rejected) path pairs."""
    acc, rej = set(), set()
    for r in brain.store.q("SELECT payload_json,status FROM proposals WHERE kind='link' AND status IN "
                           "('applied','accepted','rejected','reverted')"):
        p = json.loads(r[0])
        pair = tuple(sorted((p["src_path"], p["dst_path"])))
        (rej if r[1] in ("rejected", "reverted") else acc).add(pair)
    return acc, rej
