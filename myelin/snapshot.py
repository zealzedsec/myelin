"""A read-only snapshot of the brain: the web UI plus a JSON file, no server.

`myelin snapshot OUT --query "..."` writes a directory that opens in any static
host. It answers the preset queries with the recall results Myelin actually
produced, replays the last sleep cycle with its recorded step timings, and keeps
decisions made in the page local to the page.

Note bodies are left out unless `--with-content` is given: a snapshot is for
showing how the brain works, and titles, structure and short recall snippets are
enough for that. Whatever is included should be read before it is shared.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .brain import Brain
from .recall import RecallEngine
from .server import UI_DIR, App
from .store import atomic_write


def build_snapshot(brain: Brain, engine: RecallEngine, out: Path, queries: list[str],
                   with_content: bool = False) -> dict[str, Any]:
    app = App(brain, "127.0.0.1", 0)
    app.engine = engine
    notes: dict[str, Any] = {}
    stems: dict[str, int] = {}
    for row in brain.store.q("SELECT id FROM notes ORDER BY id"):
        n = app.note({"id": [str(row[0])]})
        stems = n.pop("stems")
        if not with_content:
            n["content"] = ""
            n["content_omitted"] = True
        notes[str(n["id"])] = n
    recalls = {}
    for q in queries:
        res = engine.recall(q, k=12, source="snapshot", log=False)
        res["recall_id"] = f"snapshot-{len(recalls) + 1}"
        recalls[q] = res
    status = app.status({})
    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # The vault's name, not its absolute path: where it lives on this machine
        # is nobody else's business once the snapshot is shared.
        "status": {**status, "root": Path(status["root"]).name, "snapshot": True,
                   "daemon_alive": False, "agent_available": False},
        "graph": app.graph({}),
        "selfmodel": app.selfmodel({}),
        "evolution": app.evolution({}),
        "metrics": app.metrics({"names": ["quiz.mrr.holdout,quiz.mrr.train"]}),
        "ledger": app.ledger({"limit": ["400"]}),
        "proposals": {s: app.proposals({"status": [s], "limit": ["300"]})["items"]
                      for s in ("pending", "applied", "rejected")},
        "notes": notes,
        "stems": stems,
        "recalls": recalls,
        "with_content": with_content,
    }
    if out.exists():
        # Only ever replace an earlier snapshot. `myelin snapshot .` must not
        # be able to delete a vault.
        if any(out.iterdir()) and not (out / "snapshot.json").is_file():
            raise ValueError(f"{out} exists and is not a previous snapshot; choose an empty or new directory")
        shutil.rmtree(out)
    shutil.copytree(UI_DIR, out)
    html = (out / "index.html").read_text(encoding="utf-8")
    html = html.replace('<meta name="myelin-token" content="%%MYELIN_TOKEN%%">',
                        '<meta name="myelin-token" content="">\n<meta name="myelin-mode" content="snapshot">')
    (out / "index.html").write_text(html, encoding="utf-8")
    atomic_write(out / "snapshot.json", json.dumps(data, default=str, separators=(",", ":")))
    return {"out": str(out), "notes": len(notes), "queries": len(recalls),
            "bytes": (out / "snapshot.json").stat().st_size, "with_content": with_content}
