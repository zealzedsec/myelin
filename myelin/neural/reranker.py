"""A learned reranker: a small MLP trained pairwise (RankNet) on what actually helped.

Training pairs come from two sources:

* **Feedback** — a person (UI) or an agent (MCP `feedback`, or reading a recalled
  note in Claude Code) marks a result useful; it is paired against results from
  the same recall that were shown above it and not used.
* **Self-quiz train items** — the gold note of a cloze item against the hard
  negatives the current ranker put above it.

The reranker starts with zero weight in the recall blend. Evolution raises
`w_rerank` only if doing so improves held-out quiz accuracy, so a reranker that
has learned nothing useful cannot degrade recall.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .autograd import Tensor, bce_with_logits
from .nn import MLP, AdamW


class Reranker:
    def __init__(self, n_features: int, seed: int = 7, hidden: int = 16):
        self.n_features = n_features
        self.seed = seed
        self.mlp = MLP([n_features, hidden, 1], np.random.default_rng(seed))
        self.mu = np.zeros(n_features)
        self.sd = np.ones(n_features)
        self.trained = False

    def _norm(self, rows: np.ndarray) -> np.ndarray:
        return (np.asarray(rows, dtype=float) - self.mu) / self.sd

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        if len(rows) == 0:
            return np.zeros(0)
        return self.mlp(Tensor(self._norm(rows))).data.ravel()

    def fit(self, pos: np.ndarray, neg: np.ndarray, epochs: int = 300, lr: float = 0.01) -> dict[str, Any]:
        pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
        if len(pos) < 12:
            return {"status": "insufficient-data", "pairs": int(len(pos))}
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(pos))
        cut = max(3, int(0.2 * len(pos)))
        val, train = order[:cut], order[cut:]
        both = np.vstack([pos[train], neg[train]])
        self.mu = both.mean(axis=0)
        self.sd = both.std(axis=0) + 1e-6
        opt = AdamW(self.mlp.parameters(), lr=lr, weight_decay=1e-3)
        ones = np.ones(len(train))
        best = (-1.0, 0, self.mlp.state())
        for epoch in range(1, epochs + 1):
            opt.zero_grad()
            diff = self.mlp(Tensor(self._norm(pos[train]))) - self.mlp(Tensor(self._norm(neg[train])))
            loss = bce_with_logits(diff, ones)
            loss.backward()
            opt.step()
            if epoch % 10 == 0:
                acc = float((self(pos[val]) > self(neg[val])).mean())
                if acc > best[0] + 1e-9:
                    best = (acc, epoch, self.mlp.state())
                elif epoch - best[1] >= 80:
                    break
        self.mlp.load_state(best[2])
        self.trained = True
        # The negatives are, by construction, notes the base ranker put *above* the
        # positive, so the base ranker's own pairwise accuracy here is near zero.
        # This is the share of the base ranker's inversions the reranker fixes.
        baseline = float((pos[val][:, 3] > neg[val][:, 3]).mean())
        return {"status": "trained", "pairs": int(len(pos)), "val_pairwise_acc": round(best[0], 4),
                "val_seed_score_acc": round(baseline, 4), "epoch": best[1]}

    def state(self) -> dict[str, np.ndarray]:
        out = self.mlp.state()
        out["mu"], out["sd"] = self.mu, self.sd
        return out

    @classmethod
    def from_state(cls, state: dict[str, np.ndarray], seed: int = 7) -> "Reranker":
        mu, sd = state["mu"], state["sd"]
        model = cls(len(mu), seed)
        model.mlp.load_state({k: v for k, v in state.items() if k.startswith("p")})
        model.mu, model.sd = mu, sd
        model.trained = True
        return model


class RerankerEnsemble:
    """Several rerankers trained on the same pairs from different seeds, scored
    by their mean. A single seed is a lottery: early stopping on a few dozen
    validation pairs settles anywhere from epoch 10 to 120, and on the
    AGENT_BRAIN_VAULT self-quiz five seeds gave held-out MRR from 0.232 to 0.256.
    Three disjoint five-seed ensembles landed at 0.249, 0.255 and 0.263."""

    def __init__(self, n_features: int, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)):
        self.members = [Reranker(n_features, seed=s) for s in seeds]

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        if len(rows) == 0:
            return np.zeros(0)
        return np.mean([m(rows) for m in self.members], axis=0)

    def fit(self, pos: np.ndarray, neg: np.ndarray) -> dict[str, Any]:
        results = [m.fit(pos, neg) for m in self.members]
        if any(r["status"] != "trained" for r in results):
            return results[0]
        return {"status": "trained", "pairs": results[0]["pairs"], "members": len(results),
                "val_pairwise_acc": round(float(np.mean([r["val_pairwise_acc"] for r in results])), 4),
                "val_seed_score_acc": round(float(np.mean([r["val_seed_score_acc"] for r in results])), 4),
                "epochs": [r["epoch"] for r in results]}

    def state(self) -> dict[str, np.ndarray]:
        return {f"m{i}.{k}": v for i, m in enumerate(self.members) for k, v in m.state().items()}


def reranker_from_state(state: dict[str, np.ndarray], seed: int = 7) -> Reranker | RerankerEnsemble:
    """Load either format: an ensemble's keys are prefixed `m0.`, `m1.`, ..."""
    if "mu" in state:
        return Reranker.from_state(state, seed)
    count = 1 + max(int(k.split(".", 1)[0][1:]) for k in state)
    ens = RerankerEnsemble.__new__(RerankerEnsemble)
    ens.members = [Reranker.from_state({k.split(".", 1)[1]: v for k, v in state.items() if k.startswith(f"m{i}.")})
                   for i in range(count)]
    return ens
