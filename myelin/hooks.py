"""Claude Code hook entry points. Each reads the hook payload on stdin.

    prompt  UserPromptSubmit — inject the few notes that clearly match the prompt
    read    PostToolUse(Read) — a vault note was read: implicit feedback + co-access
    stop    Stop             — fold the session's co-access into Hebbian edges, re-ingest

Two rules are load-bearing, learned from the vault's own `evolve/capture.py`:
a hook must **always exit 0** (a failing hook can interrupt the session), and
must print **nothing** unless it means to — UserPromptSubmit stdout is injected
into the model's context, so stray output costs tokens on every prompt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import load_config


def main(event: str, root: Path | None = None) -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    try:
        cfg = load_config(root or _root_from(payload))
        if not cfg.db_path.exists():
            return 0  # never initialise a brain from inside a hook
        from .brain import Brain
        brain = Brain(cfg)
        try:
            if event == "prompt":
                text = _prompt_context(brain, payload)
                if text:
                    sys.stdout.write(text)
            elif event == "read":
                _on_read(brain, payload)
            elif event == "stop":
                from .learning import consolidate_sessions
                brain.ingest()
                consolidate_sessions(brain)
        finally:
            brain.close()
    except Exception:
        pass  # silent by design; see module docstring
    return 0


def _root_from(payload: dict) -> Path | None:
    cwd = payload.get("cwd")
    return Path(cwd) if cwd else None


def _prompt_context(brain, payload: dict) -> str:
    cfg = brain.cfg
    prompt = (payload.get("prompt") or "").strip()
    if not cfg.inject_context or prompt.startswith("/") or len(prompt.split()) < 4:
        return ""
    from .consolidate import load_reranker
    from .recall import RecallEngine
    engine = RecallEngine(brain)
    load_reranker(brain, engine)
    res = engine.recall(prompt[:2000], k=cfg.inject_max_notes, source="hook",
                        session=payload.get("session_id"))
    if res["best_similarity"] < cfg.inject_min_score:
        return ""
    lines = [f"<myelin_recall id=\"{res['recall_id']}\" best_similarity=\"{res['best_similarity']}\">",
             "Vault notes that match this prompt (ranked by Myelin). Read any that are relevant:"]
    from .tools import defuse
    for r in res["results"][: cfg.inject_max_notes]:
        snippet = " ".join(r["snippet"].split())[:180]
        lines.append(defuse(f"- {r['path']} — {r['title']}: {snippet}", "myelin_recall"))
    lines.append("</myelin_recall>")
    return "\n".join(lines) + "\n"


def _on_read(brain, payload: dict) -> None:
    if payload.get("tool_name") != "Read":
        return
    path = (payload.get("tool_input") or {}).get("file_path")
    session = payload.get("session_id")
    if not path or not session or not path.endswith(".md"):
        return
    try:
        rel = Path(path).resolve().relative_to(brain.cfg.root.resolve()).as_posix()
    except ValueError:
        return
    from .learning import record_access
    record_access(brain, session, rel)
