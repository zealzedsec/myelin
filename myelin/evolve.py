"""Self-evolution: champion/challenger search over the recall genome.

A (1+λ) evolution strategy. Each generation mutates the current champion λ
times, scores every challenger on the **train** half of the self-quiz, and sends
only the best one to the **holdout** half. The challenger is promoted only if it
beats the champion on holdout by a margin and does not lose recall@10 — the same
gate a careful engineer would apply to a config change, run automatically.

What keeps this from fooling itself:

* selection uses train items, promotion uses holdout items; holdout is touched
  once per generation and the count of touches is recorded, because adaptive
  reuse of a holdout set slowly overfits it;
* a holdout gain has to be real, not just positive: champion and challenger
  are compared item by item, and a paired sign-flip test must put the gain
  below `ALPHA` chance. A bare mean gain of +0.005 on 48 items is noise, and a
  gate that accepts noise promotes some of it every time it is consulted;
* below `MIN_HOLDOUT` items no promotion happens at all — the report says
  "insufficient evidence" rather than promoting on noise;
* every generation, promoted or not, is a ledger event with both fitnesses and
  the exact parameter diff, so any change in behaviour can be traced and undone
  (`myelin evolve --rollback`).
"""
from __future__ import annotations

import json
import time
from dataclasses import fields
from typing import Any

import numpy as np

from .brain import Brain
from .config import GENOME_BOUNDS, RecallParams
from .metrics import paired_sign_flip_p
from .selfeval import QuizRunner
from .store import canonical_json

MIN_HOLDOUT = 15
MIN_GAIN = 0.005
ALPHA = 0.05


def _to_unit(p: RecallParams) -> np.ndarray:
    return np.array([(getattr(p, n) - lo) / (hi - lo) for n, (lo, hi, _) in GENOME_BOUNDS.items()])


def _from_unit(u: np.ndarray) -> RecallParams:
    values = {}
    for x, (name, (lo, hi, _)) in zip(np.clip(u, 0, 1), GENOME_BOUNDS.items()):
        values[name] = lo + float(x) * (hi - lo)
    return RecallParams.from_dict(values)


def mutate(p: RecallParams, sigma: float, rng: np.random.Generator, rate: float = 0.35) -> RecallParams:
    u = _to_unit(p)
    mask = rng.random(len(u)) < rate
    if not mask.any():
        mask[rng.integers(len(u))] = True
    u = u + mask * rng.normal(0.0, sigma, size=len(u))
    return _from_unit(u)


def probes(champ: RecallParams, gen: int) -> tuple[str, list[RecallParams]]:
    """Coordinate probes: one gene per generation, cycling through the genome,
    tried at its lower bound and halfway to its upper bound. Gaussian steps
    search near the champion, and once sigma has shrunk after a run of
    rejections they cannot reach a signal switched off entirely, which is just
    what the self-model's ablations sometimes say would help. The probes go
    through the same train selection and paired holdout gate as everything else."""
    names = list(GENOME_BOUNDS)
    name = names[(gen - 1) % len(names)]
    lo, hi, _ = GENOME_BOUNDS[name]
    current = float(getattr(champ, name))
    out = [RecallParams.from_dict({**champ.to_dict(), name: v}) for v in (lo, (current + hi) / 2)
           if abs(current - v) > 1e-6]
    return name, out


def diff(a: RecallParams, b: RecallParams) -> dict[str, list[float]]:
    out = {}
    for f in fields(RecallParams):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if abs(float(x) - float(y)) > 1e-9:
            out[f.name] = [round(float(x), 4), round(float(y), 4)]
    return out


def ensure_champion(brain: Brain, runner: QuizRunner) -> tuple[int, RecallParams, dict[str, Any]]:
    version, champ = brain.champion()
    if version:
        row = brain.store.one("SELECT fitness_json FROM configs WHERE version=?", (version,))
        return version, champ, json.loads(row[0])
    fitness = {"train": runner.evaluate(champ, "train"), "holdout": runner.evaluate(champ, "holdout")}
    with brain.store.tx() as db:
        db.execute("INSERT INTO configs(version,params_json,parent,status,fitness_json,created_at) "
                   "VALUES(1,?,NULL,'champion',?,?)", (canonical_json(champ.to_dict()), canonical_json(fitness),
                                                      time.time()))
    brain.store.event("CONFIG_SEEDED", "recall", {"version": 1, "fitness": fitness})
    return 1, champ, fitness


def generation(brain: Brain, runner: QuizRunner, lam: int = 8, seed: int | None = None) -> dict[str, Any]:
    version, champ, _ = ensure_champion(brain, runner)
    # Re-measure the champion on today's notes: the vault changed since it was crowned.
    champ_train = runner.evaluate(champ, "train")
    champ_hold = runner.evaluate(champ, "holdout", keep=True)
    champ_rr = champ_hold.pop("rr")
    champ_hold.pop("ranked", None)
    gen = int(brain.store.get_meta("evolution_generation", "0") or 0) + 1
    sigma = float(brain.store.get_meta("evolution_sigma", "0.15") or 0.15)
    rng = np.random.default_rng((seed if seed is not None else brain.cfg.seed) * 1_000_003 + gen)

    probed, probe_set = probes(champ, gen)
    challengers = [mutate(champ, sigma, rng) for _ in range(lam)] + probe_set
    scored = [(runner.evaluate(c, "train"), c) for c in challengers]
    best_train, best = max(scored, key=lambda t: (t[0]["mrr"], t[0]["r10"]))
    result: dict[str, Any] = {
        "generation": gen, "sigma": round(sigma, 4), "champion_version": version,
        "champion": {"train": champ_train, "holdout": champ_hold},
        "best_challenger_train": best_train, "diff": diff(champ, best), "promoted": False, "probed": probed,
    }
    holdout_n = champ_hold["n"]
    if holdout_n < MIN_HOLDOUT:
        result["verdict"] = f"insufficient evidence: {holdout_n} holdout items (< {MIN_HOLDOUT})"
    elif best_train["mrr"] <= champ_train["mrr"] + 1e-4:
        result["verdict"] = "no challenger beat the champion on train"
    else:
        best_hold = runner.evaluate(best, "holdout", keep=True)
        best_rr = best_hold.pop("rr")
        best_hold.pop("ranked", None)
        p_value = paired_sign_flip_p(np.array(best_rr) - np.array(champ_rr), rng)
        best_hold["p_value"] = round(p_value, 4)
        result["best_challenger_holdout"] = best_hold
        uses = int(brain.store.get_meta("holdout_uses", "0") or 0) + 1
        brain.store.set_meta("holdout_uses", str(uses))
        result["holdout_uses"] = uses
        gain = best_hold["mrr"] - champ_hold["mrr"]
        if gain >= MIN_GAIN and p_value <= ALPHA and best_hold["r10"] >= champ_hold["r10"] - 0.01:
            result["promoted"] = True
            result["verdict"] = f"promoted: holdout MRR +{gain:.4f} (paired p={p_value:.3f})"
        elif gain >= MIN_GAIN and p_value > ALPHA:
            result["verdict"] = f"rejected on holdout: MRR {gain:+.4f} could be chance (paired p={p_value:.3f})"
        else:
            result["verdict"] = f"rejected on holdout: MRR {gain:+.4f}, R@10 {best_hold['r10'] - champ_hold['r10']:+.4f}"

    new_version = None
    with brain.store.tx() as db:
        next_version = db.execute("SELECT COALESCE(MAX(version),0)+1 FROM configs").fetchone()[0]
        fitness = {"train": best_train, "holdout": result.get("best_challenger_holdout")}
        status = "champion" if result["promoted"] else "rejected"
        if result["promoted"]:
            db.execute("UPDATE configs SET status='retired' WHERE status='champion'")
            new_version = next_version
        db.execute("INSERT INTO configs(version,params_json,parent,status,fitness_json,created_at) VALUES(?,?,?,?,?,?)",
                   (next_version, canonical_json(best.to_dict()), version, status, canonical_json(fitness), time.time()))
    # 1/5th-success-rule flavour: widen after a win, narrow after a loss.
    sigma = min(0.5, sigma * 1.3) if result["promoted"] else max(0.03, sigma * 0.93)
    brain.store.set_meta("evolution_sigma", str(sigma))
    brain.store.set_meta("evolution_generation", str(gen))
    result["new_version"] = new_version
    brain.store.metric("quiz.mrr.holdout", champ_hold["mrr"] if not result["promoted"]
                       else result["best_challenger_holdout"]["mrr"], {"generation": gen})
    brain.store.metric("quiz.mrr.train", champ_train["mrr"] if not result["promoted"] else best_train["mrr"],
                       {"generation": gen})
    brain.store.event("EVOLUTION", "recall", {
        "generation": gen, "promoted": result["promoted"], "verdict": result["verdict"], "probed": probed,
        "champion_version": version, "new_version": new_version, "diff": result["diff"],
        "champion_holdout_mrr": round(champ_hold["mrr"], 4),
        "p_value": (result.get("best_challenger_holdout") or {}).get("p_value"),
        "challenger_holdout_mrr": round(result.get("best_challenger_holdout", {}).get("mrr", float("nan")), 4)
        if result.get("best_challenger_holdout") else None,
    })
    return result


def rollback(brain: Brain) -> dict[str, Any]:
    """Restore the previous champion. A rollback is itself a ledger event."""
    current = brain.store.one("SELECT version,parent FROM configs WHERE status='champion' ORDER BY version DESC LIMIT 1")
    if current is None or current["parent"] is None:
        return {"rolled_back": False, "reason": "no earlier champion"}
    with brain.store.tx() as db:
        db.execute("UPDATE configs SET status='retired' WHERE version=?", (current["version"],))
        db.execute("UPDATE configs SET status='champion' WHERE version=?", (current["parent"],))
    brain.store.event("CONFIG_ROLLBACK", "recall", {"from": current["version"], "to": current["parent"]})
    return {"rolled_back": True, "from": current["version"], "to": current["parent"]}
