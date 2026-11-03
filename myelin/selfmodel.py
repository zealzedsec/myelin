"""The self-model: what the brain knows, how well it recalls, and what it lacks.

"Self-aware" is used here in a narrow, checkable sense. The brain keeps a model
of itself built only from measurements: its topics (communities), its accuracy
on its own quiz with a confidence interval, which of its signals actually help
(ablations), how its link suggestions fare against human decisions
(calibration), what it was asked and could not answer (gaps), and the integrity
of its own history (ledger verification). Nothing in it is asserted; every
number is recomputed each cycle and every claim links to the data behind it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .brain import Brain
from .metrics import expected_calibration_error
from .recall import RecallEngine
from .selfeval import QuizRunner, ablations
from .store import atomic_write


def build_self_model(brain: Brain, engine: RecallEngine, runner: QuizRunner) -> dict[str, Any]:
    from .consolidate import structure

    store = brain.store
    version, champ = brain.champion()
    st = structure(brain)
    notes = {r["id"]: dict(r) for r in store.q("SELECT id,path,stem,title,type,words,mtime,frontmatter_error FROM notes")}
    graph = brain.graph()

    communities: dict[int, list[int]] = {}
    for nid, c in st["community"].items():
        if nid in notes:
            communities.setdefault(c, []).append(nid)
    topics = []
    for c, members in sorted(communities.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            continue
        members.sort(key=lambda n: -st["pagerank"].get(n, 0))
        topics.append({"id": c, "label": st["labels"].get(c, f"cluster {c}"), "size": len(members),
                       "core": [notes[n]["title"] for n in members[:4]], "core_stems": [notes[n]["stem"] for n in members[:4]]})

    quiz = {"items": len(runner.items), "holdout_items": len(runner.split("holdout"))}
    if runner.items:
        quiz["train"] = runner.evaluate(champ, "train")
        quiz["holdout"] = runner.with_ci(champ, "holdout", seed=brain.cfg.seed)
        quiz["ablations"] = ablations(runner, champ)
        quiz["random_mrr"] = round(_random_mrr(len(notes)), 4)

    edges = {r[0]: r[1] for r in store.q("SELECT kind, COUNT(*) FROM edges GROUP BY kind")}
    authored = {n for pair in graph.explicit_pairs for n in pair}
    orphans = [notes[n]["title"] for n in notes if n not in authored]
    broken = store.q("SELECT target_raw, COUNT(*) c FROM links WHERE target_path IS NULL GROUP BY target_raw "
                     "ORDER BY c DESC, target_raw LIMIT 12")
    broken_total = store.one("SELECT COUNT(*) FROM links WHERE target_path IS NULL")[0]
    link_total = store.one("SELECT COUNT(*) FROM links")[0]

    decisions = store.q("SELECT score, status FROM proposals WHERE kind='link' AND status IN "
                        "('applied','accepted','rejected','reverted')")
    calibration: dict[str, Any] = {"decided": len(decisions)}
    if decisions:
        probs = np.array([d[0] for d in decisions])
        labels = np.array([1.0 if d[1] in ("applied", "accepted") else 0.0 for d in decisions])
        calibration.update({"accept_rate": round(float(labels.mean()), 4),
                            "ece": round(expected_calibration_error(probs, labels), 4)})

    ok, msg, n_events = store.verify_ledger()
    feedback = store.one("SELECT COUNT(*), COALESCE(SUM(label>=0.5),0) FROM feedback")
    pending = {r[0]: r[1] for r in store.q("SELECT kind, COUNT(*) FROM proposals WHERE status='pending' GROUP BY kind")}
    gens = int(store.get_meta("evolution_generation", "0") or 0)
    promotions = store.one("SELECT COUNT(*) FROM configs WHERE parent IS NOT NULL AND status IN ('champion','retired')")[0]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "identity": {
            "notes": len(notes), "passages": store.one("SELECT COUNT(*) FROM passages")[0],
            "edges": edges, "encoder": store.get_meta("encoder_id"), "topics": topics[:12],
        },
        "reliability": {
            "quiz": quiz,
            "link_prediction": json.loads(store.get_meta("link_report", "{}") or "{}"),
            "reranker": _active_model(brain, "reranker"),
            "link_calibration": calibration,
        },
        "evolution": {
            "champion_version": version, "params": champ.to_dict(), "generations": gens,
            "promotions": promotions, "sigma": float(store.get_meta("evolution_sigma", "0.15") or 0.15),
            "holdout_uses": int(store.get_meta("holdout_uses", "0") or 0),
            "history": store.metric_series("quiz.mrr.holdout", 60),
        },
        "learning": {
            "feedback": feedback[0], "positive_feedback": feedback[1],
            "hebbian_edges": edges.get("hebbian", 0),
            "recalls": store.one("SELECT COUNT(*) FROM recalls")[0],
        },
        "gaps": [{"query": r["example"], "count": r["count"], "best_similarity": round(r["best_score"], 3)}
                 for r in store.q("SELECT * FROM gaps ORDER BY count DESC, last_seen DESC LIMIT 10")],
        "health": {
            "orphans": orphans[:20], "orphan_count": len(orphans),
            "broken_links": broken_total, "links": link_total,
            "broken_targets": [{"target": r[0], "count": r[1]} for r in broken],
            "duplicate_stems": json.loads(store.get_meta("duplicate_stems", "{}") or "{}"),
            "frontmatter_errors": [n["path"] for n in notes.values() if n["frontmatter_error"]],
        },
        "pending_proposals": pending,
        "ledger": {"ok": ok, "message": msg, "events": n_events},
    }


def _random_mrr(n: int, k: int = 10) -> float:
    """Expected MRR@k of a uniformly random ranking over n-1 candidates."""
    m = max(1, n - 1)
    return sum(1.0 / r for r in range(1, min(k, m) + 1)) / m


def _active_model(brain: Brain, name: str) -> dict[str, Any]:
    row = brain.store.one("SELECT version, metrics_json, created_at FROM models WHERE name=? AND active=1 "
                          "ORDER BY version DESC LIMIT 1", (name,))
    if row is None:
        return {"status": "untrained"}
    return {"version": row[0], **json.loads(row[1]), "trained_at": row[2]}


def publish(brain: Brain, model: dict[str, Any]) -> dict[str, Any]:
    atomic_write(brain.cfg.state / "self_model.json", json.dumps(model, indent=2, default=str))
    written = None
    if brain.cfg.report_note:
        path = brain.cfg.root / brain.cfg.report_note
        atomic_write(path, render_markdown(brain, model))
        written = brain.cfg.report_note
    return {"json": str(brain.cfg.state / "self_model.json"), "note": written}


def render_markdown(brain: Brain, m: dict[str, Any]) -> str:
    """A vault note. Wikilinks only to in-scope notes with unique stems, so the
    note never adds a broken or ambiguous link to the graph it describes."""
    dupes = set(m["health"]["duplicate_stems"])
    link = lambda stem, title=None: (f"[[{stem}]]" if stem not in dupes else (title or stem))
    q = m["reliability"]["quiz"]
    now = datetime.now(timezone.utc)
    lines = [
        "---",
        "type: analytics",
        "node_id: myelin_self_model",
        f"updated: {now.date().isoformat()}",
        f'generated_at: "{m["generated_at"]}"',
        "tags: [analytics, myelin, generated]",
        "---",
        "",
        "# Myelin Self-Model",
        "",
        "> Generated by Myelin's sleep cycle (`myelin sleep`); rewritten on every run, so do not",
        "> edit it here. Every number below is measured, not asserted. Home: [[00_HOME_MOC]].",
        "",
        "## How well I recall",
        "",
    ]
    if q.get("holdout"):
        h, t = q["holdout"], q["train"]
        lines += [
            f"Self-quiz: {q['items']} cloze items built from authored links ({q['holdout_items']} held out).",
            "Each item masks one `[[link]]` and asks recall to find the target from the surrounding text.",
            "",
            "| Split | n | MRR@10 | Recall@5 | Recall@10 |",
            "|---|---|---|---|---|",
            f"| train | {t['n']} | {t['mrr']:.3f} | {t['r5']:.3f} | {t['r10']:.3f} |",
            f"| holdout | {h['n']} | {h['mrr']:.3f} ({h['mrr_ci'][0]:.3f}–{h['mrr_ci'][1]:.3f}) | {h['r5']:.3f} | {h['r10']:.3f} |",
            f"| random ranking | — | {q['random_mrr']:.3f} | — | — |",
            "",
            "Signal ablations (MRR, all items): "
            + ", ".join(f"{k.replace('_', ' ')} {v:.3f}" for k, v in q["ablations"].items()) + ".",
            "",
        ]
    else:
        lines += ["No self-quiz items yet: the vault has no authored links with enough surrounding text.", ""]
    lp = m["reliability"]["link_prediction"]
    if lp.get("status") == "trained":
        lines += ["## Link prediction", "",
                  f"Chosen method: **{lp['chosen']}** (selected on validation AUC; {lp['edges']} links, {lp['nodes']} notes).", "",
                  "| Method | val AUC | test AUC | test AP |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['val_auc']:.3f} | {v['test_auc']:.3f} | {v['test_ap']:.3f} |" for k, v in lp["scores"].items()]
        lines.append("")
    elif lp:
        lines += ["## Link prediction", "", f"Not trained: {lp.get('status')}.", ""]
    ev = m["evolution"]
    lines += ["## Evolution", "",
              f"Champion recall config v{ev['champion_version']} after {ev['generations']} generations "
              f"({ev['promotions']} promotions, holdout consulted {ev['holdout_uses']} times).", ""]
    if m["identity"]["topics"]:
        lines += ["## What I know (topic communities)", ""]
        for t in m["identity"]["topics"]:
            core = ", ".join(link(s, n) for s, n in zip(t["core_stems"], t["core"]))
            lines.append(f"- **{t['label']}** — {t['size']} notes; core: {core}")
        lines.append("")
    if m["gaps"]:
        lines += ["## What I was asked and could not answer", ""]
        lines += [f"- “{g['query']}” — asked {g['count']}×, best similarity {g['best_similarity']}" for g in m["gaps"]]
        lines.append("")
    hl = m["health"]
    lines += ["## Health", "",
              f"- {hl['broken_links']} of {hl['links']} links unresolved; {hl['orphan_count']} notes with no authored links.",
              f"- Duplicate stems: {len(hl['duplicate_stems'])}; frontmatter errors: {len(hl['frontmatter_errors'])}.",
              f"- Ledger: {m['ledger']['message']}.", ""]
    return "\n".join(lines)
