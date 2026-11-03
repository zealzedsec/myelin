"""Link prediction: a graph autoencoder that learns which notes *should* be linked.

Model (Kipf & Welling's GAE, with a richer decoder):

    encoder   Z = Â · relu(Â · X · W1) · W2           two GCN layers
    decoder   p(i~j) = σ(MLP([z_i ⊙ z_j, |z_i − z_j|, s_ij]))

`X` is the LSA vector of each note plus its log-degree; `s_ij` are classic
structural signals (common neighbours, Adamic–Adar, Jaccard, cosine). Putting the
heuristics *inside* the decoder means the network only has to learn what they
miss.

Evaluation follows the standard protocol and is the part that keeps this honest:
existing links are split into train/validation/test; message passing only ever
sees train links; negatives are sampled non-links. The model is compared on the
same split against Adamic–Adar and plain cosine similarity, and the brain uses
whichever wins on validation. On a small, sparse vault a heuristic often wins —
the report says so rather than shipping the neural net regardless.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..metrics import average_precision, roc_auc
from .autograd import SparseAdj, Tensor, absolute, bce_with_logits, concat, dropout, relu, take_rows
from .nn import MLP, AdamW, GCNLayer, Module

N_STRUCT = 4


def _pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_edges(pairs: list[tuple[int, int]], n: int, rng: np.random.Generator,
                val_frac: float = 0.10, test_frac: float = 0.15) -> Split:
    """Hold out edges without isolating nodes: an edge is only held out if both
    endpoints keep at least one training edge, otherwise the encoder would see a
    held-out node with no neighbourhood at all."""
    pairs = [_pair_key(i, j) for i, j in pairs]
    order = rng.permutation(len(pairs))
    degree = np.zeros(n, dtype=int)
    for i, j in pairs:
        degree[i] += 1
        degree[j] += 1
    n_val = int(round(val_frac * len(pairs)))
    n_test = int(round(test_frac * len(pairs)))
    held: list[int] = []
    for idx in order:
        if len(held) >= n_val + n_test:
            break
        i, j = pairs[idx]
        if degree[i] > 1 and degree[j] > 1:
            degree[i] -= 1
            degree[j] -= 1
            held.append(int(idx))
    held_set = set(held)
    train = [pairs[i] for i in range(len(pairs)) if i not in held_set]
    val = [pairs[i] for i in held[:n_val]]
    test = [pairs[i] for i in held[n_val:]]
    as_arr = lambda xs: np.array(xs, dtype=np.int64).reshape(-1, 2)
    return Split(as_arr(train), as_arr(val), as_arr(test))


def sample_negatives(n: int, count: int, known: set[tuple[int, int]], rng: np.random.Generator) -> np.ndarray:
    out: set[tuple[int, int]] = set()
    max_pairs = n * (n - 1) // 2 - len(known)
    count = min(count, max(0, max_pairs))
    attempts = 0
    while len(out) < count and attempts < count * 50:
        attempts += 1
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        key = _pair_key(int(i), int(j))
        if key not in known and key not in out:
            out.add(key)
    return np.array(sorted(out), dtype=np.int64).reshape(-1, 2)


def sample_two_hop(nbrs: list[set[int]], positives: np.ndarray, known: set[tuple[int, int]],
                   rng: np.random.Generator) -> np.ndarray:
    """For each held-out link (i, j), one non-link (i, k) with k two hops from i
    in the training graph. Random non-links are mostly far apart and easy to
    reject; a pair that shares a neighbour is the case a link suggester actually
    faces, and the one neighbourhood-overlap heuristics cannot separate."""
    out: set[tuple[int, int]] = set()
    for i, j in positives:
        for a in (int(i), int(j)):
            two: set[int] = set()
            for m in nbrs[a]:
                two |= nbrs[m]
            two -= nbrs[a]
            two.discard(a)
            pool = [k for k in sorted(two) if _pair_key(a, k) not in known and _pair_key(a, k) not in out]
            if pool:
                out.add(_pair_key(a, int(rng.choice(pool))))
                break
    return np.array(sorted(out), dtype=np.int64).reshape(-1, 2)


class Structure:
    """Neighbourhood heuristics over a fixed (training) edge set."""

    def __init__(self, n: int, pairs: np.ndarray, x: np.ndarray):
        self.nbrs: list[set[int]] = [set() for _ in range(n)]
        for i, j in pairs:
            self.nbrs[int(i)].add(int(j))
            self.nbrs[int(j)].add(int(i))
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.xn = x / norms

    def features(self, pairs: np.ndarray) -> np.ndarray:
        out = np.zeros((len(pairs), N_STRUCT))
        for row, (i, j) in enumerate(pairs):
            a, b = self.nbrs[int(i)], self.nbrs[int(j)]
            common = a & b
            union = a | b
            aa = sum(1.0 / math.log(len(self.nbrs[c]) + 1.0) for c in common if len(self.nbrs[c]) > 0)
            out[row] = (math.log1p(len(common)), aa, len(common) / len(union) if union else 0.0,
                        float(self.xn[int(i)] @ self.xn[int(j)]))
        return out


class GraphAutoEncoder(Module):
    def __init__(self, in_dim: int, rng: np.random.Generator, hidden: int = 64, out: int = 32,
                 dropout_rate: float = 0.2):
        self.g1 = GCNLayer(in_dim, hidden, rng)
        self.g2 = GCNLayer(hidden, out, rng)
        self.dec = MLP([2 * out + N_STRUCT, 32, 1], rng)
        self.rate = dropout_rate

    def encode(self, adj: SparseAdj, x: np.ndarray, rng: np.random.Generator, training: bool) -> Tensor:
        h = relu(self.g1(adj, dropout(Tensor(x), self.rate, rng, training)))
        return self.g2(adj, dropout(h, self.rate, rng, training))

    def logits(self, z: Tensor, pairs: np.ndarray, struct: np.ndarray) -> Tensor:
        zi, zj = take_rows(z, pairs[:, 0]), take_rows(z, pairs[:, 1])
        return self.dec(concat([zi * zj, absolute(zi - zj), Tensor(struct)], axis=1))


@dataclass
class LinkReport:
    status: str
    chosen: str = "none"
    nodes: int = 0
    edges: int = 0
    epochs: int = 0
    seconds: float = 0.0
    scores: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "chosen": self.chosen, "nodes": self.nodes, "edges": self.edges,
                "epochs": self.epochs, "seconds": self.seconds, "scores": self.scores}


def _features(x: np.ndarray, pairs: np.ndarray, n: int) -> np.ndarray:
    degree = np.zeros(n)
    for i, j in pairs:
        degree[i] += 1
        degree[j] += 1
    deg = np.log1p(degree)[:, None]
    return np.hstack([x, deg / max(deg.max(), 1.0)])


def _fit(model: GraphAutoEncoder, adj: SparseAdj, feats: np.ndarray, train: np.ndarray, known: set,
         struct: Structure, rng: np.random.Generator, epochs: int, lr: float,
         val: tuple[np.ndarray, np.ndarray] | None = None, patience: int = 40,
         hard_negatives: np.ndarray | None = None) -> tuple[int, float]:
    """Pairs a person rejected are the most informative negatives there are, so
    every epoch trains on them alongside freshly sampled random non-links."""
    opt = AdamW(model.parameters(), lr=lr)
    train_struct = struct.features(train)
    best_auc, best_epoch, best_state = -1.0, 0, model.state()
    hard = hard_negatives if hard_negatives is not None else np.zeros((0, 2), dtype=np.int64)
    for epoch in range(1, epochs + 1):
        neg = np.vstack([sample_negatives(adj.n, len(train), known, rng), hard])
        pairs = np.vstack([train, neg])
        labels = np.concatenate([np.ones(len(train)), np.zeros(len(neg))])
        s = np.vstack([train_struct, struct.features(neg)])
        opt.zero_grad()
        z = model.encode(adj, feats, rng, training=True)
        loss = bce_with_logits(model.logits(z, pairs, s), labels)
        loss.backward()
        opt.step()
        if val is not None and (epoch % 5 == 0 or epoch == epochs):
            auc = roc_auc(_predict(model, adj, feats, val[0], struct, rng), val[1])
            if auc > best_auc + 1e-4:
                best_auc, best_epoch, best_state = auc, epoch, model.state()
            elif epoch - best_epoch >= patience:
                break
    if val is not None:
        model.load_state(best_state)
        return best_epoch, best_auc
    return epochs, float("nan")


def _predict(model: GraphAutoEncoder, adj: SparseAdj, feats: np.ndarray, pairs: np.ndarray,
             struct: Structure, rng: np.random.Generator) -> np.ndarray:
    z = model.encode(adj, feats, rng, training=False)
    return model.logits(z, pairs, struct.features(pairs)).data.ravel()


class LinkPredictor:
    """Trains, evaluates against baselines, selects, and scores candidate links."""

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.model: GraphAutoEncoder | None = None
        self.report = LinkReport("untrained")
        self._adj: SparseAdj | None = None
        self._feats: np.ndarray | None = None
        self._struct: Structure | None = None

    def train(self, x: np.ndarray, pairs: list[tuple[int, int]], rejected: set[tuple[int, int]] | None = None,
              max_epochs: int = 300, lr: float = 0.01) -> LinkReport:
        started = time.time()
        n = x.shape[0]
        uniq = sorted({_pair_key(i, j) for i, j in pairs if i != j})
        rep = LinkReport("insufficient-data", nodes=n, edges=len(uniq))
        if n < 8 or len(uniq) < 20:
            self.report = rep
            return rep
        rng = np.random.default_rng(self.seed)
        split = split_edges(uniq, n, rng)
        if len(split.val) < 3 or len(split.test) < 3:
            self.report = rep
            return rep
        known = set(uniq) | set(rejected or ())
        val_neg = sample_negatives(n, len(split.val), known, rng)
        test_neg = sample_negatives(n, len(split.test), known | {tuple(p) for p in val_neg}, rng)
        val_pairs = np.vstack([split.val, val_neg])
        val_y = np.concatenate([np.ones(len(split.val)), np.zeros(len(val_neg))])
        test_pairs = np.vstack([split.test, test_neg])
        test_y = np.concatenate([np.ones(len(split.test)), np.zeros(len(test_neg))])

        feats = _features(x, split.train, n)
        struct = Structure(n, split.train, x)
        adj = SparseAdj.normalized(n, [tuple(p) for p in split.train])
        hard = np.array(sorted(_pair_key(i, j) for i, j in (rejected or ()) if i != j),
                        dtype=np.int64).reshape(-1, 2)
        model = GraphAutoEncoder(feats.shape[1], np.random.default_rng(self.seed))
        # Training negatives may only avoid what training is allowed to know: the
        # training links and the pairs a person rejected. Excluding validation and
        # test links too would tell the model which unseen pairs are real, an
        # advantage the untrained baselines do not get.
        known_train = {_pair_key(int(a), int(b)) for a, b in split.train} | set(rejected or ())
        best_epoch, _ = _fit(model, adj, feats, split.train, known_train, struct, rng, max_epochs, lr,
                             val=(val_pairs, val_y), hard_negatives=hard)

        scores: dict[str, dict[str, float]] = {}
        val_struct, test_struct = struct.features(val_pairs), struct.features(test_pairs)
        candidates = {
            "gae": (_predict(model, adj, feats, val_pairs, struct, rng),
                    _predict(model, adj, feats, test_pairs, struct, rng)),
            "adamic_adar": (val_struct[:, 1] + 1e-3 * val_struct[:, 3], test_struct[:, 1] + 1e-3 * test_struct[:, 3]),
            "cosine": (val_struct[:, 3], test_struct[:, 3]),
        }
        for name, (v, t) in candidates.items():
            scores[name] = {"val_auc": round(roc_auc(v, val_y), 4), "test_auc": round(roc_auc(t, test_y), 4),
                            "test_ap": round(average_precision(t, test_y), 4)}
        # The same test links against two-hop non-links: reported, never used to choose.
        used = known | {tuple(map(int, q)) for q in val_neg} | {tuple(map(int, q)) for q in test_neg}
        near = sample_two_hop(struct.nbrs, split.test, used, rng)
        if len(near) >= 3:
            near_pairs = np.vstack([split.test, near])
            near_y = np.concatenate([np.ones(len(split.test)), np.zeros(len(near))])
            ns = struct.features(near_pairs)
            near_scores = {"gae": _predict(model, adj, feats, near_pairs, struct, rng),
                           "adamic_adar": ns[:, 1] + 1e-3 * ns[:, 3], "cosine": ns[:, 3]}
            for name, sc in near_scores.items():
                scores[name]["test_auc_2hop"] = round(roc_auc(sc, near_y), 4)
        chosen = max(scores, key=lambda k: (scores[k]["val_auc"], k == "gae"))

        # Production fit: same recipe on every known link, for the chosen epoch count.
        full = np.array(uniq, dtype=np.int64)
        self._feats = _features(x, full, n)
        self._struct = Structure(n, full, x)
        self._adj = SparseAdj.normalized(n, uniq)
        if chosen == "gae":
            self.model = GraphAutoEncoder(self._feats.shape[1], np.random.default_rng(self.seed))
            _fit(self.model, self._adj, self._feats, full, known, self._struct,
                 np.random.default_rng(self.seed + 1), max(best_epoch, 20), lr, hard_negatives=hard)
        rep = LinkReport("trained", chosen, n, len(uniq), best_epoch, round(time.time() - started, 2), scores)
        self.report = rep
        return rep

    def score(self, pairs: np.ndarray) -> np.ndarray:
        """Comparable scores in [0, 1] for candidate pairs under the chosen method."""
        if self._struct is None or len(pairs) == 0:
            return np.zeros(len(pairs))
        s = self._struct.features(pairs)
        if self.report.chosen == "gae" and self.model is not None:
            logits = _predict(self.model, self._adj, self._feats, pairs, self._struct,
                              np.random.default_rng(0))
            return 1.0 / (1.0 + np.exp(-logits))
        if self.report.chosen == "adamic_adar":
            return 1.0 - np.exp(-s[:, 1])
        return np.clip(s[:, 3], 0.0, 1.0)

    def evidence(self, i: int, j: int) -> dict[str, Any]:
        if self._struct is None:
            return {}
        # Most specific first: a shared low-degree neighbour explains a link; a
        # shared home MOC explains nothing.
        common = sorted(self._struct.nbrs[i] & self._struct.nbrs[j],
                        key=lambda c: (len(self._struct.nbrs[c]), c))
        return {"common_neighbors": common[:6], "cosine": round(float(self._struct.xn[i] @ self._struct.xn[j]), 4)}

    def candidates(self, per_node: int = 12) -> np.ndarray:
        """Unlinked pairs worth scoring: each node's nearest neighbours in content
        space plus its two-hop graph neighbourhood."""
        if self._struct is None:
            return np.zeros((0, 2), dtype=np.int64)
        xn = self._struct.xn
        n = xn.shape[0]
        out: set[tuple[int, int]] = set()
        k = min(per_node, n - 1)
        for start in range(0, n, 512):          # blockwise: never materialise n×n
            sims = xn[start:start + 512] @ xn.T
            for r in range(sims.shape[0]):
                sims[r, start + r] = -np.inf
            nearest = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            for r, row in enumerate(nearest):
                for j in row:
                    out.add(_pair_key(start + r, int(j)))
        for i in range(n):
            for mid in self._struct.nbrs[i]:
                for j in list(self._struct.nbrs[mid])[:per_node]:
                    if j != i:
                        out.add(_pair_key(i, j))
        existing = {_pair_key(i, j) for i in range(n) for j in self._struct.nbrs[i]}
        return np.array(sorted(out - existing), dtype=np.int64).reshape(-1, 2)

    def save_state(self) -> dict[str, np.ndarray]:
        return self.model.state() if self.model is not None else {}
