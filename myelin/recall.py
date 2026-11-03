"""Recall: from a question to a ranked, explained set of notes.

Pipeline, split so the self-quiz can cache the expensive half:

1. **Candidates** (config-independent): dense cosine over passage vectors and
   BM25 over the FTS5 index, each aggregated to the note level by max.
2. **Ranking** (config-dependent, cheap): seed score = blend of the two signals;
   the top `seed_k` seeds spread activation through the graph; the final score
   adds activation and, once it has earned a non-zero weight, the learned
   reranker's score.

Every result carries its score breakdown and, if it was reached through the
graph, the seed and edge kind that delivered it.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .brain import Brain
from .config import RecallParams
from .text import fts_query, tokenize

FEATURES = ["dense", "sparse", "activation", "seed", "title_overlap", "log_degree",
            "recency", "hebbian", "is_hub_type"]


@dataclass
class Candidates:
    query: str
    dense: dict[int, float]
    sparse: dict[int, float]
    passage: dict[int, tuple[int, float]]   # note -> (best passage id, its score)
    tokens: set[str] = field(default_factory=set)


@dataclass
class Hit:
    note_id: int
    score: float
    parts: dict[str, float]
    via: dict[str, Any] | None = None
    passage_id: int | None = None


class RecallEngine:
    def __init__(self, brain: Brain):
        self.brain = brain
        self.reranker: Callable[[np.ndarray], np.ndarray] | None = None
        self._reranker_key: tuple[int, str] | None = None
        self._note_meta: tuple[str, dict[int, dict[str, Any]]] | None = None

    def sync_reranker(self) -> bool:
        """Follow the active reranker. A sleep in another thread or process may
        have trained a newer one, and the champion's `w_rerank` was tuned against
        that one, so a long-lived engine (web server, MCP server) must not keep
        scoring with the model it loaded at start. Cheap when nothing changed."""
        row = self.brain.store.one("SELECT version, path, sha256 FROM models WHERE name='reranker' AND active=1 "
                                   "ORDER BY version DESC LIMIT 1")
        key = (int(row["version"]), row["sha256"]) if row else None
        if key == self._reranker_key:
            return self.reranker is not None
        if row is None:
            self.reranker, self._reranker_key = None, None
            return False
        from .neural.nn import load_weights
        from .neural.reranker import reranker_from_state
        from .store import sha256_file
        path = Path(row["path"])
        if not path.exists() or sha256_file(path) != row["sha256"]:
            return self.reranker is not None   # a file that does not match its digest is never loaded
        self.reranker = reranker_from_state(load_weights(path), seed=self.brain.cfg.seed)
        self._reranker_key = key
        return True

    # ── stage 1 ──────────────────────────────────────────────────────────
    def candidates(self, query: str, pool: int = 60) -> Candidates:
        dense_scores: dict[int, float] = {}
        best_passage: dict[int, tuple[int, float]] = {}
        d = self.brain.dense()
        enc = self.brain.encoder()
        if enc is not None and len(d["passage_ids"]):
            q = enc.encode([query])[0]
            sims = d["passages"] @ q
            top = np.argsort(-sims)[: pool * 3]
            for i in top:
                nid, s = int(d["passage_note"][i]), float(sims[i])
                if s > dense_scores.get(nid, -1.0):
                    dense_scores[nid] = s
                    best_passage[nid] = (int(d["passage_ids"][i]), s)
        sparse_scores: dict[int, float] = {}
        match = fts_query(query)
        if match:
            rows = self.brain.store.q(
                "SELECT p.id, p.note_id, bm25(passages_fts, 2.0, 1.0) AS s FROM passages_fts "
                "JOIN passages p ON p.id = passages_fts.rowid WHERE passages_fts MATCH ? "
                "ORDER BY s LIMIT ?", (match, pool * 3))
            for r in rows:
                s = -float(r["s"])     # FTS5 bm25: more negative is better
                if s > sparse_scores.get(r["note_id"], -1e9):
                    sparse_scores[r["note_id"]] = s
                    if r["note_id"] not in best_passage:
                        best_passage[r["note_id"]] = (int(r["id"]), 0.0)
        return Candidates(query, _top(dense_scores, pool), _top(sparse_scores, pool), best_passage,
                          set(tokenize(query)))

    # ── stage 2 ──────────────────────────────────────────────────────────
    def rank(self, c: Candidates, p: RecallParams, k: int = 10,
             exclude: set[int] | frozenset = frozenset(),
             masked: set[tuple[int, int]] | frozenset = frozenset()) -> list[Hit]:
        dense = _unit(c.dense, floor=0.0)
        sparse = _unit(c.sparse, floor=0.0)
        seed_score: dict[int, float] = {}
        for nid in set(dense) | set(sparse):
            if nid in exclude:
                continue
            seed_score[nid] = p.w_dense * dense.get(nid, 0.0) + p.w_sparse * sparse.get(nid, 0.0)
        ranked_seeds = sorted(seed_score.items(), key=lambda kv: -kv[1])[: p.seed_k]
        top_seed = ranked_seeds[0][1] if ranked_seeds else 1.0
        seeds = {n: s / top_seed for n, s in ranked_seeds if top_seed > 0}

        graph = self.brain.graph()
        activation, reasons = graph.spread(seeds, p, masked=masked, blocked=exclude)
        # Raw, not max-normalised: seeds are already on a 0–1 scale, so raw spread
        # is in comparable units, and a weak spread stays weak instead of being
        # inflated to full strength for whichever node happened to receive most.
        act = activation

        pool = (set(seed_score) | set(act)) - set(exclude)
        hits: list[Hit] = []
        meta = self.note_meta()
        hebb = self._hebbian_to_seeds(seeds, graph)
        rows: list[list[float]] = []
        for nid in pool:
            parts = {
                "dense": dense.get(nid, 0.0),
                "sparse": sparse.get(nid, 0.0),
                "seed": seed_score.get(nid, 0.0),
                "activation": act.get(nid, 0.0),
            }
            hits.append(Hit(nid, parts["seed"] + p.w_activation * parts["activation"], parts,
                            passage_id=c.passage.get(nid, (None, 0))[0]))
            if self.reranker is not None and p.w_rerank > 0:
                rows.append(self._feature_row(nid, parts, c, meta, graph, hebb))
        if rows and self.reranker is not None:
            learned = self.reranker(np.asarray(rows))
            learned = (learned - learned.min()) / (np.ptp(learned) + 1e-9)
            for hit, s in zip(hits, learned):
                hit.parts["rerank"] = float(s)
                hit.score += p.w_rerank * float(s)
        hits.sort(key=lambda h: (-h.score, h.note_id))
        hits = hits[:k]
        for hit in hits:
            reason = reasons.get(hit.note_id)
            if reason and hit.parts["activation"] > 0.05:
                hit.via = {"seed": reason.seed, "kind": reason.via, "hops": reason.hops}
        return hits

    def features_for(self, c: Candidates, p: RecallParams, note_ids: list[int],
                     exclude: set[int] | frozenset = frozenset(),
                     masked: set[tuple[int, int]] | frozenset = frozenset()) -> np.ndarray:
        """Feature rows for arbitrary notes under one query (reranker training)."""
        saved, self.reranker = self.reranker, None
        try:
            hits = {h.note_id: h for h in self.rank(c, p, k=10 ** 6, exclude=exclude, masked=masked)}
        finally:
            self.reranker = saved
        graph, meta = self.brain.graph(), self.note_meta()
        dense, sparse = _unit(c.dense, 0.0), _unit(c.sparse, 0.0)
        seeds = {h.note_id: h.parts["seed"] for h in sorted(hits.values(), key=lambda h: -h.parts["seed"])[: p.seed_k]}
        hebb = self._hebbian_to_seeds(seeds, graph)
        out = []
        for nid in note_ids:
            parts = hits[nid].parts if nid in hits else {
                "dense": dense.get(nid, 0.0), "sparse": sparse.get(nid, 0.0), "seed": 0.0, "activation": 0.0}
            out.append(self._feature_row(nid, parts, c, meta, graph, hebb))
        return np.asarray(out, dtype=np.float64)

    def _feature_row(self, nid: int, parts: dict[str, float], c: Candidates, meta: dict[int, dict[str, Any]],
                     graph, hebb: dict[int, float]) -> list[float]:
        m = meta.get(nid, {})
        title_tokens = m.get("title_tokens", set())
        overlap = len(title_tokens & c.tokens) / max(1, len(title_tokens)) if title_tokens else 0.0
        return [
            parts["dense"], parts["sparse"], parts["activation"], parts["seed"], overlap,
            float(np.log1p(graph.degree(nid))), m.get("recency", 0.0), hebb.get(nid, 0.0),
            1.0 if m.get("type") in ("moc", "index", "hub") else 0.0,
        ]

    def _hebbian_to_seeds(self, seeds: dict[int, float], graph) -> dict[int, float]:
        out: dict[int, float] = {}
        for s in seeds:
            for v, w, kind, _ in graph.neighbors(s):
                if kind == "hebbian":
                    out[v] = max(out.get(v, 0.0), w)
        return out

    def note_meta(self) -> dict[int, dict[str, Any]]:
        version = self.brain.store.get_meta("graph_version", "0") or "0"
        if self._note_meta and self._note_meta[0] == version:
            return self._note_meta[1]
        now = time.time()
        meta = {}
        for r in self.brain.store.q("SELECT id,title,type,mtime FROM notes"):
            age_days = max(0.0, (now - r["mtime"]) / 86400.0)
            meta[r["id"]] = {"title_tokens": set(tokenize(r["title"])), "type": r["type"],
                             "recency": float(np.exp(-age_days / 90.0))}
        self._note_meta = (version, meta)
        return meta

    # ── full call with logging ───────────────────────────────────────────
    def recall(self, query: str, k: int = 8, source: str = "api", session: str | None = None,
               params: RecallParams | None = None, log: bool = True) -> dict[str, Any]:
        k = max(1, int(k))
        self.sync_reranker()
        version, champion = self.brain.champion()
        p = params or champion
        started = time.time()
        c = self.candidates(query)
        hits = self.rank(c, p, k=k)
        payload = self.render(query, hits, c)
        payload["config_version"] = version
        payload["ms"] = round((time.time() - started) * 1000, 1)
        # Seed scores are normalised per query, so the top hit always looks
        # perfect. Confidence has to come from the raw cosine instead.
        best = max(c.dense.values(), default=0.0)
        payload["best_similarity"] = round(best, 4)
        payload["confident"] = best >= self.brain.cfg.gap_threshold
        if log:
            rid = uuid.uuid4().hex[:12]
            payload["recall_id"] = rid
            self.brain.store.x(
                "INSERT INTO recalls(id,ts,query,source,session,config_version,results_json) VALUES(?,?,?,?,?,?,?)",
                (rid, time.time(), query, source, session, version,
                 json.dumps([{"note_id": h.note_id, "score": h.score, "parts": h.parts} for h in hits])))
            if not payload["confident"]:
                self._record_gap(query, best)
        return payload

    def render(self, query: str, hits: list[Hit], c: Candidates | None = None) -> dict[str, Any]:
        store = self.brain.store
        titles = {r["id"]: (r["title"], r["path"], r["stem"]) for r in store.q("SELECT id,title,path,stem FROM notes")}
        out = []
        for h in hits:
            title, path, stem = titles.get(h.note_id, ("?", "?", "?"))
            snippet = ""
            if h.passage_id:
                row = store.one("SELECT heading,text FROM passages WHERE id=?", (h.passage_id,))
                if row:
                    snippet = _snippet(row["text"], c.tokens if c else set())
                    heading = row["heading"]
                else:
                    heading = title
            else:
                heading = title
            item = {"note_id": h.note_id, "title": title, "path": path, "stem": stem, "heading": heading,
                    "score": round(h.score, 4), "parts": {k: round(v, 4) for k, v in h.parts.items()},
                    "snippet": snippet}
            if h.via:
                seed = titles.get(h.via["seed"], ("?", "?", "?"))
                item["via"] = {"seed_id": h.via["seed"], "seed_title": seed[0], "kind": h.via["kind"],
                               "hops": h.via["hops"]}
            out.append(item)
        return {"query": query, "results": out}

    def _record_gap(self, query: str, best: float) -> None:
        norm = " ".join(sorted(set(tokenize(query))))[:200]
        if not norm:
            return
        self.brain.store.x(
            "INSERT INTO gaps(query_norm,example,count,best_score,last_seen) VALUES(?,?,1,?,?) "
            "ON CONFLICT(query_norm) DO UPDATE SET count=count+1, best_score=max(best_score,excluded.best_score), "
            "last_seen=excluded.last_seen", (norm, query[:300], best, time.time()))


def _top(scores: dict[int, float], n: int) -> dict[int, float]:
    return dict(sorted(scores.items(), key=lambda kv: -kv[1])[:n])


def _unit(scores: dict[int, float], floor: float) -> dict[int, float]:
    if not scores:
        return {}
    top = max(scores.values())
    if top <= floor:
        return {k: 0.0 for k in scores}
    return {k: max(0.0, (v - floor) / (top - floor)) for k, v in scores.items()}


def _plain(text: str) -> str:
    """Markdown syntax out, words in: snippets are read, not rendered."""
    text = re.sub(r"\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?", " ", text)   # table rules
    text = re.sub(r"\s*\|\s*", " · ", text)
    text = re.sub(r"(\*\*|__|`|~~)", "", text)
    text = re.sub(r"(^|\s)#{1,6}\s+", r"\1", text)
    text = re.sub(r"(?:^|\s)(?:[-*+]|\d+\.)\s+(?=\S)", " ", text)
    text = re.sub(r"(?:\s*·\s*){2,}", " · ", text)
    return " ".join(text.split()).strip(" ·")


def _snippet(text: str, tokens: set[str], width: int = 260) -> str:
    text = _plain(text)
    if len(text) <= width:
        return text
    lower = text.lower()
    best, best_hits = 0, -1
    for start in range(0, max(1, len(text) - width), 40):
        window = lower[start:start + width]
        hits = sum(1 for t in tokens if re.search(r"\b" + re.escape(t), window))
        if hits > best_hits:
            best, best_hits = start, hits
    prefix = "… " if best > 0 else ""
    suffix = " …" if best + width < len(text) else ""
    return prefix + text[best:best + width].strip() + suffix
