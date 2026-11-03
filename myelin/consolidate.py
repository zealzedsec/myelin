"""Sleep: the consolidation cycle that turns a day of use into a better brain.

Named for what it imitates — offline memory consolidation. One cycle:

 1. ingest changed notes (refitting the semantic space if the corpus drifted)
 2. consolidate session co-activations into Hebbian edges; prune decayed ones
 3. recompute structure: PageRank importance and communities
 4. train the link predictor; stage its best suggestions as proposals
 5. train the reranker on feedback and quiz pairs, if there are enough
 6. run evolution generations on the self-quiz
 7. stage review proposals: near-duplicate notes and recurring knowledge gaps
 8. rebuild the self-model and publish it

Every step is timed and its outcome lands in one `SLEEP` ledger event. Steps
fail independently: a broken step is reported, not allowed to stop the cycle.
"""
from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import locks
from . import proposals as P
from .brain import Brain
from .evolve import generation
from .learning import consolidate_sessions, prune_hebbian
from .neural.linkpred import LinkPredictor
from .neural.nn import save_weights
from .neural.reranker import RerankerEnsemble
from .recall import FEATURES, RecallEngine
from .selfeval import QuizRunner, build_quiz
from .store import canonical_json, sha256_file
from .text import tokenize

MAX_LINK_PROPOSALS = 25   # new suggestions per cycle
# The inbox stops growing at this many undecided link suggestions. Without a
# ceiling, nightly sleep adds 25 a night to an inbox nobody is triaging, and a
# list that only grows is one people stop reading.
MAX_PENDING_LINKS = 60


class SleepBusy(RuntimeError):
    """Another process is already consolidating this vault."""


def sleep(brain: Brain, engine: RecallEngine | None = None, generations: int = 3,
          on_step: Callable[[str, dict[str, Any]], None] | None = None) -> dict[str, Any]:
    # One cycle per vault at a time, across the CLI, the daemon and the server:
    # two concurrent cycles would train over each other's model files.
    hold = locks.try_lock(brain.cfg.state / "sleep.lock")
    if hold is None:
        raise SleepBusy("another Myelin process is consolidating this vault right now")
    try:
        return _sleep(brain, engine, generations, on_step)
    finally:
        locks.release(hold)


def _sleep(brain: Brain, engine: RecallEngine | None, generations: int,
           on_step: Callable[[str, dict[str, Any]], None] | None) -> dict[str, Any]:
    engine = engine or RecallEngine(brain)
    load_reranker(brain, engine)
    report: dict[str, Any] = {"steps": {}, "started": time.time()}
    runner: dict[str, QuizRunner] = {}

    def step(name: str, fn: Callable[[], Any]) -> None:
        t = time.time()
        try:
            out = fn()
            entry = {"ok": True, "seconds": round(time.time() - t, 2), "result": out}
        except Exception as exc:  # one failed step must not abort consolidation
            entry = {"ok": False, "seconds": round(time.time() - t, 2), "error": f"{type(exc).__name__}: {exc}",
                     "trace": traceback.format_exc(limit=3)}
        report["steps"][name] = entry
        if on_step:
            on_step(name, entry)

    def quiz() -> QuizRunner:
        if "r" not in runner:
            runner["r"] = QuizRunner(engine, build_quiz(engine))
        return runner["r"]

    step("ingest", lambda: brain.ingest().__dict__)
    step("hebbian", lambda: {**consolidate_sessions(brain), "pruned": prune_hebbian(brain)})
    step("structure", lambda: compute_structure(brain))
    step("link_prediction", lambda: train_links(brain))
    step("reranker", lambda: train_reranker(brain, engine, quiz()))
    step("evolution", lambda: [_brief(generation(brain, quiz(), lam=brain.cfg.evolution_lambda))
                               for _ in range(generations)])
    step("review", lambda: {"duplicates": propose_duplicates(brain), "gaps": propose_gaps(brain)})

    from .selfmodel import build_self_model, publish
    step("self_model", lambda: publish(brain, build_self_model(brain, engine, quiz())))

    ok, msg, _ = brain.store.verify_ledger()
    report["ledger"] = {"ok": ok, "message": msg}
    report["seconds"] = round(time.time() - report["started"], 2)
    summary = {name: ({"ok": e["ok"], "seconds": e["seconds"]} if e["ok"] else {"ok": False, "error": e["error"]})
               for name, e in report["steps"].items()}
    brain.store.event("SLEEP", "brain", {"seconds": report["seconds"], "steps": summary})
    brain.store.set_meta("last_sleep", str(time.time()))
    return report


def _brief(gen: dict[str, Any]) -> dict[str, Any]:
    return {"generation": gen["generation"], "promoted": gen["promoted"], "verdict": gen["verdict"],
            "diff": gen["diff"] if gen["promoted"] else {}}


# ── structure ────────────────────────────────────────────────────────────
def compute_structure(brain: Brain) -> dict[str, Any]:
    g = brain.graph()
    rank = g.pagerank()
    comm = g.communities()
    labels = community_labels(brain, comm)
    brain.store.set_meta("structure", canonical_json({
        "pagerank": {str(k): round(v, 6) for k, v in rank.items()},
        "community": {str(k): v for k, v in comm.items()},
        "labels": {str(k): v for k, v in labels.items()},
    }))
    sizes: dict[int, int] = {}
    for c in comm.values():
        sizes[c] = sizes.get(c, 0) + 1
    return {"communities": len(sizes), "largest": max(sizes.values()) if sizes else 0,
            "labels": {str(k): v for k, v in list(labels.items())[:8]}}


def community_labels(brain: Brain, comm: dict[int, int]) -> dict[int, str]:
    """Name each community by the words that are both common inside it and rare
    outside it: per-note presence, scored as share-in-cluster × log(lift).
    Titles count three times, tags twice, headings once."""
    doc_terms: dict[int, dict[str, float]] = {}
    for r in brain.store.q("SELECT n.id, n.title, n.tags_json, GROUP_CONCAT(p.heading, ' | ') h FROM notes n "
                           "LEFT JOIN passages p ON p.note_id=n.id GROUP BY n.id"):
        weights: dict[str, float] = {}
        for text, w in ((r["title"], 3.0), (" ".join(json.loads(r["tags_json"])), 2.0),
                        (" ".join(part.split(" › ")[-1] for part in (r["h"] or "").split(" | ")), 1.0)):
            for tok in tokenize(text):
                if len(tok) > 2 and not tok.isdigit() and "." not in tok:
                    weights[tok] = max(weights.get(tok, 0.0), w)
        doc_terms[r["id"]] = weights
    n_docs = max(1, len(doc_terms))
    overall: dict[str, float] = {}
    for weights in doc_terms.values():
        for tok, w in weights.items():
            overall[tok] = overall.get(tok, 0.0) + w
    members: dict[int, list[int]] = {}
    for nid, c in comm.items():
        if nid in doc_terms:
            members.setdefault(c, []).append(nid)
    out: dict[int, str] = {}
    for c, ids in members.items():
        local: dict[str, float] = {}
        for nid in ids:
            for tok, w in doc_terms[nid].items():
                local[tok] = local.get(tok, 0.0) + w
        size = len(ids)
        scored = []
        for tok, v in local.items():
            share = v / size
            lift = share / (overall[tok] / n_docs)
            if v >= min(2.0, size) and lift > 1.0:
                scored.append((share * math.log(lift), tok))
        scored.sort(reverse=True)
        picked: list[str] = []
        for _, tok in scored:
            # "self-evolution", "self", "evolution" is one idea, not three.
            if any(tok in p or p in tok for p in picked):
                continue
            picked.append(tok)
            if len(picked) == 3:
                break
        out[c] = " · ".join(picked) or f"cluster {c}"
    return out


def structure(brain: Brain) -> dict[str, Any]:
    raw = brain.store.get_meta("structure")
    if not raw:
        return {"pagerank": {}, "community": {}, "labels": {}}
    data = json.loads(raw)
    return {
        "pagerank": {int(k): v for k, v in data["pagerank"].items()},
        "community": {int(k): v for k, v in data["community"].items()},
        "labels": {int(k): v for k, v in data["labels"].items()},
    }


# ── link prediction ──────────────────────────────────────────────────────
def train_links(brain: Brain) -> dict[str, Any]:
    nm = brain.note_matrix()
    ids, x = nm["ids"], nm["matrix"].astype(np.float64)
    if len(ids) < 2:
        return {"status": "insufficient-data"}
    pos = {nid: i for i, nid in enumerate(ids)}
    pairs = [(pos[a], pos[b]) for a, b in brain.graph().undirected_pairs() if a in pos and b in pos]
    paths = {r["id"]: r["path"] for r in brain.store.q("SELECT id,path FROM notes")}
    by_path = {v: k for k, v in paths.items()}
    _, rejected_paths = P.link_labels(brain)
    rejected = {(pos[by_path[a]], pos[by_path[b]]) for a, b in rejected_paths
                if a in by_path and b in by_path and by_path[a] in pos and by_path[b] in pos}

    lp = LinkPredictor(seed=brain.cfg.seed)
    rep = lp.train(x, pairs, rejected)
    out = rep.to_dict()
    brain.store.set_meta("link_report", canonical_json(out))
    if rep.status != "trained":
        return out
    if rep.chosen == "gae" and lp.model is not None:
        path = brain.cfg.models_dir / "linkpred.npz"
        save_weights(path, lp.save_state())
        digest = sha256_file(path)
        version = brain.store.one("SELECT COALESCE(MAX(version),0)+1 FROM models WHERE name='linkpred'")[0]
        with brain.store.tx() as db:
            db.execute("UPDATE models SET active=0 WHERE name='linkpred'")
            db.execute("INSERT INTO models(name,version,path,sha256,metrics_json,created_at,active) VALUES(?,?,?,?,?,?,1)",
                       ("linkpred", version, str(path), digest, canonical_json(rep.scores), time.time()))
        out["sha256"] = digest
    brain.store.event("MODEL_TRAINED", "linkpred", {"chosen": rep.chosen, "scores": rep.scores,
                                                    "edges": rep.edges, "sha256": out.get("sha256")})
    for name, s in rep.scores.items():
        brain.store.metric(f"linkpred.{name}.test_auc", s["test_auc"])

    cands = lp.candidates()
    scores = lp.score(cands)
    deg = np.array([len(lp._struct.nbrs[i]) for i in range(len(ids))], dtype=float)
    # Hub discount. Any link predictor learns preferential attachment: "link it
    # to the home MOC" is almost always *likely* and almost never *useful*. Rank
    # by probability divided by how ordinary the target's popularity makes it.
    median = max(1.0, float(np.median(deg)))
    hubness = np.sqrt(np.maximum(1.0, np.maximum(deg[cands[:, 0]], deg[cands[:, 1]]) / median)) \
        if len(cands) else np.zeros(0)
    utility = scores / hubness if len(cands) else scores
    order = np.argsort(-utility)
    titles = {r["id"]: r["title"] for r in brain.store.q("SELECT id,title FROM notes")}
    waiting = brain.store.one("SELECT COUNT(*) FROM proposals WHERE kind='link' AND status='pending'")[0]
    budget = max(0, min(MAX_LINK_PROPOSALS, MAX_PENDING_LINKS - waiting))
    out["pending_before"], out["budget"] = waiting, budget
    added, per_src, per_dst = 0, {}, {}
    for k in order:
        if added >= budget:
            break
        i, j = (int(v) for v in cands[k])
        a, b = ids[i], ids[j]
        src, dst = (a, b) if deg[i] <= deg[j] else (b, a)
        if P.is_protected(brain.cfg, paths[src]):
            src, dst = dst, src
            if P.is_protected(brain.cfg, paths[src]):
                continue
        if per_src.get(src, 0) >= 2 or per_dst.get(dst, 0) >= 2:
            continue
        ev = lp.evidence(pos[src], pos[dst])
        common = [titles.get(ids[c], "?") for c in ev.get("common_neighbors", [])][:3]
        why = (f"related through {', '.join(common)}" if common else
               f"similar content (cosine {ev.get('cosine', 0):.2f})")
        payload = {"src_path": paths[src], "dst_path": paths[dst], "src_title": titles[src],
                   "dst_title": titles[dst], "why": why, "probability": round(float(scores[k]), 4),
                   "utility": round(float(utility[k]), 4), "evidence": {**ev, "common_titles": common}}
        if P.upsert(brain, "link", P.link_key(paths[src], paths[dst]), payload, float(scores[k]), rep.chosen):
            added += 1
        per_src[src] = per_src.get(src, 0) + 1
        per_dst[dst] = per_dst.get(dst, 0) + 1
    out["proposed"] = added
    return out


# ── reranker ─────────────────────────────────────────────────────────────
def training_pairs(brain: Brain, engine: RecallEngine, runner: QuizRunner) -> tuple[np.ndarray, np.ndarray, dict]:
    _, p = brain.champion()
    base = type(p).from_dict({**p.to_dict(), "w_rerank": 0.0})
    pos_rows, neg_rows = [], []
    counts = {"quiz": 0, "feedback": 0}
    for i in runner.split("train"):
        it = runner.items[i]
        excl, mask = {it.src}, {(it.src, it.gold), (it.gold, it.src)}
        hits = engine.rank(runner.cands[i], base, k=30, exclude=excl, masked=mask)
        ranked = [h.note_id for h in hits]
        if it.gold not in ranked:
            continue
        above = [n for n in ranked[: ranked.index(it.gold)]][:3] or [n for n in ranked if n != it.gold][:1]
        if not above:
            continue
        rows = engine.features_for(runner.cands[i], base, [it.gold, *above], exclude=excl, masked=mask)
        for r in rows[1:]:
            pos_rows.append(rows[0])
            neg_rows.append(r)
            counts["quiz"] += 1
    for rec in brain.store.q("SELECT r.id, r.query, r.results_json FROM recalls r WHERE EXISTS "
                             "(SELECT 1 FROM feedback f WHERE f.recall_id=r.id)"):
        labels = {f[0]: f[1] for f in brain.store.q("SELECT note_id, MAX(label) FROM feedback WHERE recall_id=? "
                                                    "GROUP BY note_id", (rec["id"],))}
        shown = [item["note_id"] for item in json.loads(rec["results_json"])]
        positives = [n for n in shown if labels.get(n, 0) >= 0.5]
        negatives = [n for n in shown if n in labels and labels[n] < 0.5] or \
                    [n for n in shown if n not in labels][:2]
        if not positives or not negatives:
            continue
        c = engine.candidates(rec["query"])
        rows = engine.features_for(c, base, positives + negatives)
        for pi in range(len(positives)):
            for ni in range(len(positives), len(positives) + len(negatives)):
                pos_rows.append(rows[pi])
                neg_rows.append(rows[ni])
                counts["feedback"] += 1
    return np.asarray(pos_rows), np.asarray(neg_rows), counts


def train_reranker(brain: Brain, engine: RecallEngine, runner: QuizRunner) -> dict[str, Any]:
    pos, neg, counts = training_pairs(brain, engine, runner)
    model = RerankerEnsemble(len(FEATURES), seeds=tuple(brain.cfg.seed * 10 + i for i in range(5)))
    result = model.fit(pos, neg) if len(pos) else {"status": "insufficient-data", "pairs": 0}
    result["sources"] = counts
    if result["status"] != "trained":
        return result
    # Champion/challenger for the model too. A retrain is not automatically an
    # improvement: early stopping on a few dozen validation pairs can settle at
    # epoch 120 one night and epoch 10 the next, and replacing the better model
    # unconditionally cost a quarter of holdout MRR once. The candidate goes live
    # only if recall on the train items is at least as good with it. Train items,
    # because the holdout is reserved for evolution's promotion gate.
    engine.sync_reranker()
    incumbent = engine.reranker
    if incumbent is not None:
        _, p = brain.champion()
        probe = type(p).from_dict({**p.to_dict(), "w_rerank": max(p.w_rerank, 0.25)})
        active_mrr = runner.evaluate(probe, "train")["mrr"]
        engine.reranker = model
        candidate_mrr = runner.evaluate(probe, "train")["mrr"]
        engine.reranker = incumbent
        result["train_mrr"] = {"active": round(active_mrr, 4), "candidate": round(candidate_mrr, 4)}
        if candidate_mrr < active_mrr:
            result["status"] = "kept-previous"
            brain.store.event("MODEL_REJECTED", "reranker", {k: v for k, v in result.items() if k != "status"})
            return result
    version = brain.store.one("SELECT COALESCE(MAX(version),0)+1 FROM models WHERE name='reranker'")[0]
    path = brain.cfg.models_dir / f"reranker-v{version}.npz"   # versioned, so the incumbent survives a retrain
    save_weights(path, model.state())
    digest = sha256_file(path)
    with brain.store.tx() as db:
        db.execute("UPDATE models SET active=0 WHERE name='reranker'")
        db.execute("INSERT INTO models(name,version,path,sha256,metrics_json,created_at,active) VALUES(?,?,?,?,?,?,1)",
                   ("reranker", version, str(path), digest, canonical_json(result), time.time()))
    brain.store.event("MODEL_TRAINED", "reranker", {**result, "sha256": digest, "version": version})
    brain.store.metric("reranker.val_pairwise_acc", result["val_pairwise_acc"])
    engine.reranker = model
    engine._reranker_key = (int(version), digest)
    _prune_model_files(brain, "reranker")
    return result


def _prune_model_files(brain: Brain, name: str, keep: int = 3) -> None:
    """Weights of the last few versions stay on disk (the active one always)."""
    rows = brain.store.q("SELECT version, path, active FROM models WHERE name=? ORDER BY version DESC", (name,))
    protected = {r["path"] for r in rows[:keep]} | {r["path"] for r in rows if r["active"]}
    for r in rows[keep:]:
        if r["path"] not in protected:
            Path(r["path"]).unlink(missing_ok=True)


def load_reranker(brain: Brain, engine: RecallEngine) -> bool:
    return engine.sync_reranker()


# ── review proposals ─────────────────────────────────────────────────────
def propose_duplicates(brain: Brain) -> int:
    nm = brain.note_matrix()
    ids, m = nm["ids"], nm["matrix"]
    if len(ids) < 2:
        return 0
    paths = {r["id"]: (r["path"], r["title"]) for r in brain.store.q("SELECT id,path,title FROM notes")}
    added = 0
    for start in range(0, len(ids), 512):
        sims = m[start:start + 512] @ m.T
        for r, row in enumerate(sims):
            i = start + r
            for j in np.nonzero(row >= brain.cfg.duplicate_threshold)[0]:
                if j <= i:
                    continue
                a, b = paths[ids[i]], paths[ids[int(j)]]
                payload = {"a_path": a[0], "a_title": a[1], "b_path": b[0], "b_title": b[1],
                           "similarity": round(float(row[j]), 4)}
                added += P.upsert(brain, "merge", "merge:" + "|".join(sorted((a[0], b[0]))), payload,
                                  float(row[j]), "cosine")
    return added


def propose_gaps(brain: Brain, min_count: int = 2) -> int:
    added = 0
    for r in brain.store.q("SELECT * FROM gaps WHERE count>=? ORDER BY count DESC LIMIT 20", (min_count,)):
        payload = {"example": r["example"], "count": r["count"], "best_similarity": round(r["best_score"], 4)}
        added += P.upsert(brain, "gap", "gap:" + r["query_norm"], payload, float(r["count"]), "recall-log")
    return added
