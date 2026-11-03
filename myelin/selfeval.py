"""The self-quiz: a benchmark the brain builds from its own notes.

Every authored link is a fact about what the author found relevant. A **cloze
item** takes the paragraph around a link `[[B]]` in note A, masks the link, and
asks recall to find B — "given this context, which note belongs here?". That is
exactly the question a writer asks while drafting, so the metric measures the
thing the brain is for.

Leakage controls, without which the number is meaningless:

* the source note A is excluded from candidates and blocked as an activation
  seed (it contains the answer as text);
* every edge between A and B is masked during spreading activation (the link
  being predicted must not be used to predict itself);
* items are split train/holdout by a stable hash of the *paths*, so the split
  survives re-indexing, and evolution only ever selects on train.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import RecallParams
from .metrics import bootstrap_ci, reciprocal_rank
from .recall import Candidates, RecallEngine
from .text import tokenize


@dataclass
class QuizItem:
    src: int
    gold: int
    query: str
    holdout: bool
    key: str


def build_quiz(engine: RecallEngine, max_items: int = 600, min_tokens: int = 5) -> list[QuizItem]:
    rows = engine.brain.store.q(
        "SELECT l.src, l.dst, l.context, s.path AS sp, d.path AS dp FROM links l "
        "JOIN notes s ON s.id=l.src JOIN notes d ON d.id=l.dst "
        "WHERE l.dst IS NOT NULL AND l.dst != l.src AND l.from_frontmatter=0 ORDER BY l.id")
    items: dict[str, QuizItem] = {}
    for r in rows:
        key = f"{r['sp']}->{r['dp']}"
        context = r["context"]
        if key in items or len(tokenize(context)) < min_tokens:
            continue
        h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
        items[key] = QuizItem(r["src"], r["dst"], context, h % 5 == 0, key)
    ordered = sorted(items.values(), key=lambda it: hashlib.sha1(it.key.encode()).hexdigest())
    return ordered[:max_items]


class QuizRunner:
    def __init__(self, engine: RecallEngine, items: list[QuizItem]):
        self.engine = engine
        self.items = items
        started = time.time()
        self.cands: list[Candidates] = [engine.candidates(it.query) for it in items]
        self.prep_seconds = round(time.time() - started, 2)

    def split(self, which: str) -> list[int]:
        if which == "all":
            return list(range(len(self.items)))
        want = which == "holdout"
        return [i for i, it in enumerate(self.items) if it.holdout == want]

    def evaluate(self, p: RecallParams, which: str = "train", keep: bool = False) -> dict[str, Any]:
        rr, hit5, hit10, ranks = [], [], [], []
        for i in self.split(which):
            it = self.items[i]
            hits = self.engine.rank(self.cands[i], p, k=10, exclude={it.src},
                                    masked={(it.src, it.gold), (it.gold, it.src)})
            ranked = [h.note_id for h in hits]
            r = reciprocal_rank(ranked, it.gold, 10)
            rr.append(r)
            hit5.append(1.0 if it.gold in ranked[:5] else 0.0)
            hit10.append(1.0 if it.gold in ranked[:10] else 0.0)
            if keep:
                ranks.append((i, ranked))
        n = len(rr)
        out: dict[str, Any] = {
            "n": n,
            "mrr": float(np.mean(rr)) if n else 0.0,
            "r5": float(np.mean(hit5)) if n else 0.0,
            "r10": float(np.mean(hit10)) if n else 0.0,
        }
        if keep:
            out["rr"] = rr
            out["ranked"] = ranks
        return out

    def with_ci(self, p: RecallParams, which: str, seed: int = 0) -> dict[str, Any]:
        res = self.evaluate(p, which, keep=True)
        lo, hi = bootstrap_ci(np.array(res["rr"]), np.random.default_rng(seed))
        res["mrr_ci"] = [round(lo, 4), round(hi, 4)]
        res.pop("ranked", None)
        res.pop("rr", None)
        return res


def ablations(runner: QuizRunner, p: RecallParams, which: str = "all") -> dict[str, float]:
    """What each signal contributes: MRR with one component switched off."""
    variants = {
        "full": p,
        "dense_only": RecallParams.from_dict({**p.to_dict(), "w_sparse": 0.0, "w_activation": 0.0, "w_rerank": 0.0}),
        "sparse_only": RecallParams.from_dict({**p.to_dict(), "w_dense": 0.0, "w_activation": 0.0, "w_rerank": 0.0}),
        "no_graph": RecallParams.from_dict({**p.to_dict(), "w_activation": 0.0}),
        "no_learned_edges": RecallParams.from_dict({**p.to_dict(), "k_semantic": 0.0, "k_hebbian": 0.0}),
    }
    return {name: round(runner.evaluate(v, which)["mrr"], 4) for name, v in variants.items()}
