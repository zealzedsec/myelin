"""Graph dynamics, recall, the self-quiz, evolution, and the neural models."""
import numpy as np
import pytest

from myelin import evolve as E
from myelin.config import RecallParams
from myelin.graph import Graph, hebbian_effective, hebbian_update
from myelin.neural.linkpred import LinkPredictor, split_edges
from myelin.neural.reranker import Reranker
from myelin.recall import RecallEngine
from myelin.selfeval import QuizRunner, build_quiz


def _line_graph():
    rows = [(1, 2, "explicit", 1.0, 0.0), (2, 3, "explicit", 1.0, 0.0), (1, 4, "semantic", 0.9, 0.0)]
    return Graph.from_rows([1, 2, 3, 4], rows, half_life_days=30, now=0.0)


def test_spread_follows_direction_mask_and_block():
    g = _line_graph()
    p = RecallParams(steps=2, decay=0.5, fan_exponent=0.0, k_backlink=0.0)
    act, why = g.spread({1: 1.0}, p)
    assert act[2] > 0 and act[3] > 0 and act[4] > 0
    assert why[3].seed == 1 and why[3].hops == 2
    # backlinks conduct only at k_backlink
    act_back, _ = g.spread({2: 1.0}, p)
    assert 1 not in act_back
    act_back, _ = g.spread({2: 1.0}, RecallParams(steps=1, k_backlink=0.5))
    assert act_back.get(1, 0) > 0
    masked, _ = g.spread({1: 1.0}, p, masked={(1, 2)})
    assert 2 not in masked and 3 not in masked
    blocked, _ = g.spread({1: 1.0}, p, blocked={2})
    assert 2 not in blocked and 3 not in blocked


def test_hub_fan_dilutes_activation():
    rows = [(0, i, "explicit", 1.0, 0.0) for i in range(1, 11)] + [(20, 21, "explicit", 1.0, 0.0)]
    g = Graph.from_rows([0, *range(1, 11), 20, 21], rows, 30, 0.0)
    p = RecallParams(steps=1, fan_exponent=1.0)
    hub, _ = g.spread({0: 1.0}, p)
    lone, _ = g.spread({20: 1.0}, p)
    assert hub[1] < lone[21]


def test_hebbian_rule_is_bounded_and_decays():
    w = 0.0
    for _ in range(100):
        w = hebbian_update(w, 0.3)
    assert 0.99 < w <= 1.0
    assert hebbian_effective(1.0, 0.0, 10.0, 10 * 86400) == pytest.approx(0.5)


def test_structure_pagerank_and_communities():
    rows = [(a, b, "explicit", 1.0, 0.0) for a in range(4) for b in range(4) if a < b]
    rows += [(a, b, "explicit", 1.0, 0.0) for a in range(10, 14) for b in range(10, 14) if a < b]
    g = Graph.from_rows([*range(4), *range(10, 14)], rows, 30, 0.0)
    pr = g.pagerank()
    assert sum(pr.values()) == pytest.approx(1.0)
    comm = g.communities()
    assert len({comm[i] for i in range(4)}) == 1 and comm[0] != comm[10]


def test_recall_finds_topic_and_explains_graph_hits(brain):
    eng = RecallEngine(brain)
    res = eng.recall("sourdough levain hydration proofing")
    assert res["results"] and res["results"][0]["path"].startswith("notes/BAKE_")
    assert res["confident"]
    assert all(r["parts"]["dense"] >= 0 for r in res["results"])


def test_low_confidence_queries_become_gaps(brain):
    eng = RecallEngine(brain)
    for _ in range(2):
        res = eng.recall("quantum chromodynamics lattice gauge")
    assert not res["confident"]
    assert brain.store.one("SELECT count FROM gaps")[0] == 2


def test_quiz_masks_the_link_it_tests(brain):
    eng = RecallEngine(brain)
    items = build_quiz(eng)
    assert len(items) > 50 and any(it.holdout for it in items)
    for it in items[:20]:
        assert "[...]" in it.query
    runner = QuizRunner(eng, items)
    p = brain.champion()[1]
    res = runner.evaluate(p, "all", keep=True)
    for i, ranked in res["ranked"]:
        assert items[i].src not in ranked          # the source note is never an answer
    assert 0 < res["mrr"] <= 1


def test_evolution_promotes_only_on_holdout_gain(brain, monkeypatch):
    eng = RecallEngine(brain)
    runner = QuizRunner(eng, build_quiz(eng))
    monkeypatch.setattr(E, "MIN_HOLDOUT", 10 ** 6)
    res = E.generation(brain, runner, lam=4)
    assert not res["promoted"] and "insufficient evidence" in res["verdict"]
    monkeypatch.setattr(E, "MIN_HOLDOUT", 1)
    promoted = 0
    for _ in range(12):
        res = E.generation(brain, runner, lam=6)
        if res["promoted"]:
            promoted += 1
            hold = res["best_challenger_holdout"]["mrr"]
            assert hold >= res["champion"]["holdout"]["mrr"] + E.MIN_GAIN
    kinds = [r[0] for r in brain.store.q("SELECT kind FROM events WHERE kind='EVOLUTION'")]
    assert len(kinds) == 13
    if promoted:
        before = brain.champion()[0]
        out = E.rollback(brain)
        assert out["rolled_back"] and brain.champion()[0] != before
    assert brain.store.verify_ledger()[0]


def test_mutation_stays_in_bounds():
    rng = np.random.default_rng(0)
    p = RecallParams()
    for _ in range(200):
        p = E.mutate(p, 0.5, rng)
        assert 3 <= p.seed_k <= 40 and 1 <= p.steps <= 3 and 0 <= p.k_backlink <= 1


def _planted(seed=0, n=120):
    rng = np.random.default_rng(seed)
    comm = rng.integers(0, 4, n)
    x = np.hstack([np.eye(4)[comm] + 0.6 * rng.normal(size=(n, 4)), rng.normal(size=(n, 8))])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if rng.random() < (0.15 if comm[i] == comm[j] else 0.004)]
    return x, pairs


def test_link_predictor_learns_and_is_compared_to_baselines():
    x, pairs = _planted()
    lp = LinkPredictor(seed=1)
    rep = lp.train(x, pairs)
    assert rep.status == "trained"
    assert set(rep.scores) == {"gae", "adamic_adar", "cosine"}
    assert rep.scores[rep.chosen]["val_auc"] == max(s["val_auc"] for s in rep.scores.values())
    assert rep.scores["gae"]["test_auc"] > 0.6
    cands = lp.candidates()
    existing = {tuple(sorted(p)) for p in pairs}
    assert not ({tuple(c) for c in cands.tolist()} & existing)
    s = lp.score(cands[:20])
    assert ((0 <= s) & (s <= 1)).all()


def test_link_predictor_refuses_tiny_graphs():
    rep = LinkPredictor().train(np.eye(5), [(0, 1), (1, 2)])
    assert rep.status == "insufficient-data"


def test_split_never_isolates_a_node():
    x, pairs = _planted(2, 60)
    split = split_edges(pairs, 60, np.random.default_rng(0))
    deg = np.zeros(60)
    for i, j in split.train:
        deg[i] += 1
        deg[j] += 1
    touched = {i for p in pairs for i in p}
    assert all(deg[i] > 0 for i in touched)


def test_reranker_learns_a_separable_preference():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(80, 9))
    neg = rng.normal(size=(80, 9))
    pos[:, 4] += 2.0  # the useful feature
    model = Reranker(9, seed=0)
    out = model.fit(pos, neg)
    assert out["status"] == "trained" and out["val_pairwise_acc"] > 0.8
    restored = Reranker.from_state(model.state())
    assert np.allclose(restored(pos[:5]), model(pos[:5]))
