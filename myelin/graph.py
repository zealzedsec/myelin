"""The association graph: typed weighted edges, spreading activation, structure.

Edge kinds and where they come from:

| kind       | source                                   | learned? |
|------------|------------------------------------------|----------|
| explicit   | `[[wikilinks]]` written in note bodies    | no — authored |
| related    | links inside frontmatter properties       | no — authored |
| semantic   | k-nearest neighbours in the LSA space     | yes — from content |
| hebbian    | notes used together in the same session  | yes — from behaviour |

Spreading activation follows the ACT-R / Collins–Loftus model: seed nodes get
activation from the query, and each hop passes on `decay` of it, divided by the
fan of both endpoints (`degree ** fan_exponent`, as in GCN symmetric
normalisation) so that hub notes like a home MOC do not flood every result. Every activated node remembers which seed and which
edge kind delivered most of its activation; that is the explanation shown to
the user ("reached from CLAUDE via an explicit link").
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .config import RecallParams



@dataclass
class Contribution:
    seed: int
    via: str
    hops: int


class Graph:
    def __init__(self, node_ids: list[int]):
        self.nodes = list(node_ids)
        self.pos = {n: i for i, n in enumerate(self.nodes)}
        # adjacency[u] -> list of (v, weight, kind, forward). Authored links are
        # stored both ways; the reverse copy has forward=False and conducts at
        # `k_backlink`. Learned edges are symmetric, so both copies are forward.
        self.adj: dict[int, list[tuple[int, float, str, bool]]] = defaultdict(list)
        self.out_degree: dict[int, int] = defaultdict(int)
        self.explicit_pairs: set[tuple[int, int]] = set()

    @classmethod
    def from_rows(cls, node_ids: list[int], rows: list[tuple[int, int, str, float, float]],
                  half_life_days: float, now: float | None = None) -> "Graph":
        g = cls(node_ids)
        now = now or time.time()
        merged: dict[tuple[int, int, str, bool], float] = {}
        for src, dst, kind, weight, updated_at in rows:
            if src == dst or src not in g.pos or dst not in g.pos:
                continue
            if kind == "hebbian":
                weight = hebbian_effective(weight, updated_at, half_life_days, now)
            authored = kind in ("explicit", "related")
            if authored:
                g.explicit_pairs.add((src, dst))
            for a, b, fwd in ((src, dst, True), (dst, src, not authored)):
                key = (a, b, kind, fwd)
                merged[key] = max(merged.get(key, 0.0), weight)
        for (a, b, kind, fwd), w in merged.items():
            if w > 0:
                g.adj[a].append((b, w, kind, fwd))
        for a, items in g.adj.items():
            g.out_degree[a] = len({item[0] for item in items})
        return g

    def degree(self, n: int) -> int:
        return self.out_degree.get(n, 0)

    def neighbors(self, n: int) -> list[tuple[int, float, str, bool]]:
        return self.adj.get(n, [])

    # ── spreading activation ─────────────────────────────────────────────
    def spread(self, seeds: dict[int, float], p: RecallParams,
               masked: set[tuple[int, int]] | frozenset = frozenset(),
               blocked: set[int] | frozenset = frozenset(),
               frontier_cap: int = 64) -> tuple[dict[int, float], dict[int, Contribution]]:
        conductance = {"explicit": p.k_explicit, "related": p.k_related,
                       "semantic": p.k_semantic, "hebbian": p.k_hebbian}
        activation: dict[int, float] = {}
        best: dict[int, tuple[float, Contribution]] = {}
        frontier = {n: (a, Contribution(n, "seed", 0)) for n, a in seeds.items() if n not in blocked}
        for hop in range(1, p.steps + 1):
            incoming: dict[int, float] = defaultdict(float)
            reasons: dict[int, tuple[float, Contribution]] = {}
            for u, (a_u, origin) in frontier.items():
                fan_u = max(1, self.degree(u)) ** p.fan_exponent
                for v, w, kind, forward in self.adj.get(u, ()):
                    if v in blocked or (u, v) in masked:
                        continue
                    if not forward:
                        w *= p.k_backlink
                    # Symmetric fan: a hub dilutes what it sends *and* what it
                    # absorbs, or a home MOC that every note links to wins
                    # every query.
                    fan = fan_u * max(1, self.degree(v)) ** p.fan_exponent
                    c = a_u * p.decay * conductance.get(kind, 0.0) * w / fan
                    if c <= 1e-6:
                        continue
                    incoming[v] += c
                    if c > reasons.get(v, (0.0, None))[0]:
                        seed = origin.seed if origin.hops else u
                        reasons[v] = (c, Contribution(seed, kind, hop))
            for v, a in incoming.items():
                activation[v] = activation.get(v, 0.0) + a
                if v in reasons and reasons[v][0] > best.get(v, (0.0, None))[0]:
                    best[v] = reasons[v]
            ranked = sorted(incoming.items(), key=lambda kv: -kv[1])[:frontier_cap]
            frontier = {v: (a, reasons[v][1]) for v, a in ranked if v in reasons}
            if not frontier:
                break
        return activation, {n: c for n, (_, c) in best.items()}

    # ── structure ────────────────────────────────────────────────────────
    def pagerank(self, damping: float = 0.85, iters: int = 60) -> dict[int, float]:
        n = len(self.nodes)
        if n == 0:
            return {}
        src, dst, w = [], [], []
        for u, items in self.adj.items():
            total = sum(item[1] for item in items)
            for v, x, _, _ in items:
                src.append(self.pos[u])
                dst.append(self.pos[v])
                w.append(x / total)
        src_a, dst_a, w_a = np.array(src, dtype=int), np.array(dst, dtype=int), np.array(w)
        dangling = np.ones(n, dtype=bool)
        dangling[src_a] = False
        rank = np.full(n, 1.0 / n)
        for _ in range(iters):
            nxt = np.zeros(n)
            np.add.at(nxt, dst_a, rank[src_a] * w_a)
            nxt = damping * (nxt + rank[dangling].sum() / n) + (1 - damping) / n
            if np.abs(nxt - rank).sum() < 1e-9:
                rank = nxt
                break
            rank = nxt
        return {self.nodes[i]: float(rank[i]) for i in range(n)}

    def communities(self, resolution: float = 1.0) -> dict[int, int]:
        """Louvain modularity optimisation (Blondel et al. 2008), deterministic.

        Plain label propagation put 53 of AGENT_BRAIN_VAULT's 55 notes in one
        community, because a home MOC that links everything makes the whole graph
        look like one neighbourhood. Two changes fix that: modularity rewards a
        community only for edges beyond what degree alone predicts, and each edge
        is divided by sqrt(deg_u * deg_v) so a hub's many links count for little.
        Frontmatter links, mostly pointers back to the MOC, count at 0.3."""
        kind_w = {"explicit": 1.0, "related": 0.3, "semantic": 0.6, "hebbian": 1.0}
        w: dict[tuple[int, int], float] = defaultdict(float)
        for u, items in self.adj.items():
            for v, x, kind, fwd in items:
                if u < v and (fwd or kind in ("explicit", "related")):
                    w[(u, v)] += x * kind_w.get(kind, 0.5)
        deg: dict[int, int] = defaultdict(int)
        for u, v in w:
            deg[u] += 1
            deg[v] += 1
        edges = {(u, v): x / math.sqrt(deg[u] * deg[v]) for (u, v), x in w.items()}
        part = _louvain(list(self.nodes), edges, resolution)
        sizes: dict[int, int] = defaultdict(int)
        for c in part.values():
            sizes[c] += 1
        order = sorted(sizes, key=lambda c: (-sizes[c], min(n for n, k in part.items() if k == c)))
        remap = {c: i for i, c in enumerate(order)}
        return {n: remap[c] for n, c in part.items()}

    def undirected_pairs(self, kinds: tuple[str, ...] = ("explicit", "related")) -> set[tuple[int, int]]:
        pairs = set()
        for u, items in self.adj.items():
            for v, _, kind, _ in items:
                if kind in kinds and u != v:
                    pairs.add((min(u, v), max(u, v)))
        return pairs


def hebbian_effective(weight: float, updated_at: float, half_life_days: float, now: float) -> float:
    age_days = max(0.0, (now - updated_at) / 86400.0)
    return weight * math.pow(0.5, age_days / max(half_life_days, 1e-6))


def hebbian_update(weight: float, lr: float, strength: float = 1.0) -> float:
    """Bounded Hebbian step: w ← w + lr·s·(1 − w). Saturates at 1, never overshoots."""
    return weight + lr * strength * (1.0 - weight)


def _louvain(nodes: list[int], edges: dict[tuple[int, int], float], resolution: float) -> dict[int, int]:
    """Multi-level Louvain over an undirected weighted edge map. Returns node -> community."""
    membership = {n: n for n in nodes}
    level_nodes = list(nodes)
    level_edges = dict(edges)
    while True:
        moved, assign = _louvain_level(level_nodes, level_edges, resolution)
        membership = {n: assign[c] for n, c in membership.items()}
        if not moved:
            return membership
        agg: dict[tuple[int, int], float] = defaultdict(float)
        for (u, v), x in level_edges.items():
            a, b = assign[u], assign[v]
            agg[(min(a, b), max(a, b))] += x
        level_nodes = sorted(set(assign.values()))
        level_edges = dict(agg)


def _louvain_level(nodes: list[int], edges: dict[tuple[int, int], float], resolution: float) -> tuple[bool, dict[int, int]]:
    nbrs: dict[int, dict[int, float]] = {n: {} for n in nodes}
    k: dict[int, float] = {n: 0.0 for n in nodes}
    total = 0.0
    for (u, v), x in edges.items():
        if u == v:
            k[u] += 2 * x
            total += 2 * x
            continue
        nbrs[u][v] = nbrs[u].get(v, 0.0) + x
        nbrs[v][u] = nbrs[v].get(u, 0.0) + x
        k[u] += x
        k[v] += x
        total += 2 * x
    comm = {n: n for n in nodes}
    tot = dict(k)
    if total <= 0:
        return False, comm
    moved_any = False
    for _ in range(50):
        moved = False
        for n in sorted(nodes):
            current = comm[n]
            links: dict[int, float] = defaultdict(float)
            for m, x in nbrs[n].items():
                links[comm[m]] += x
            tot[current] -= k[n]
            best, best_gain = current, links.get(current, 0.0) - resolution * tot[current] * k[n] / total
            for c in sorted(links):
                gain = links[c] - resolution * tot[c] * k[n] / total
                if gain > best_gain + 1e-12:
                    best, best_gain = c, gain
            tot[best] += k[n]
            if best != current:
                comm[n] = best
                moved = moved_any = True
        if not moved:
            break
    return moved_any, comm
