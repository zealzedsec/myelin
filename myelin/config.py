"""Configuration: defaults, `myelin.toml` loading, and the evolvable recall genome.

Two kinds of settings live here and they are deliberately kept apart:

* **Operator settings** (`Config`) — scope rules, paths, ports, budgets. Only a
  person changes these, by editing `myelin.toml`.
* **The recall genome** (`RecallParams`) — blend weights and activation dynamics.
  The evolution engine mutates these and promotes a challenger only when it beats
  the incumbent on held-out self-quiz items. A person can pin values, but the
  default is that the brain tunes them from evidence.
"""
from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

CONFIG_NAME = "myelin.toml"
STATE_DIR_NAME = ".myelin"

# Directory names that are never notes, wherever they appear.
DEFAULT_EXCLUDE_DIRS = [
    ".git", ".obsidian", ".trash", ".myelin", "node_modules", "__pycache__",
    ".pytest_cache", ".venv", "venv",
]


@dataclass
class RecallParams:
    """The evolvable genome. Bounds live in `GENOME_BOUNDS`."""

    w_dense: float = 0.55        # weight of embedding similarity in the seed score
    w_sparse: float = 0.45       # weight of BM25 lexical match in the seed score
    w_activation: float = 1.0    # weight of graph-spread activation in the final score
    w_rerank: float = 0.0        # weight of the learned reranker (0 until it earns it)
    seed_k: int = 12             # how many top candidates seed the spread
    steps: int = 2               # spreading-activation hops
    decay: float = 0.55          # activation retained per hop
    fan_exponent: float = 0.5    # ACT-R fan effect: spread / out_degree**fan_exponent
    k_explicit: float = 1.0      # per-edge-kind conductance
    k_related: float = 0.8
    k_semantic: float = 0.6
    k_hebbian: float = 1.0
    # Measured on AGENT_BRAIN_VAULT's self-quiz: spreading along backlinks cost
    # 0.044 MRR (0.322 -> 0.278), because it pulls in the index notes that link
    # to everything. Evolution may raise it on a vault where it helps.
    k_backlink: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RecallParams":
        base = cls()
        if not data:
            return base
        known = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key in known:
                kind = type(getattr(base, key))
                setattr(base, key, kind(value))
        return base.clamped()

    def clamped(self) -> "RecallParams":
        out = copy.copy(self)
        for name, (lo, hi, integer) in GENOME_BOUNDS.items():
            value = min(max(getattr(out, name), lo), hi)
            setattr(out, name, int(round(value)) if integer else float(value))
        return out


# name -> (low, high, is_integer)
GENOME_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "w_dense": (0.0, 1.0, False),
    "w_sparse": (0.0, 1.0, False),
    "w_activation": (0.0, 4.0, False),
    "w_rerank": (0.0, 1.0, False),
    "seed_k": (3, 40, True),
    "steps": (1, 3, True),
    "decay": (0.15, 0.95, False),
    "fan_exponent": (0.0, 1.0, False),
    "k_explicit": (0.0, 1.5, False),
    "k_related": (0.0, 1.5, False),
    "k_semantic": (0.0, 1.5, False),
    "k_hebbian": (0.0, 2.0, False),
    "k_backlink": (0.0, 1.0, False),
}


@dataclass
class Config:
    root: Path
    include: list[str] = field(default_factory=lambda: ["**/*.md"])
    exclude: list[str] = field(default_factory=list)
    exclude_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_DIRS))
    exclude_files: list[str] = field(default_factory=list)
    # Notes another program regenerates. Myelin reads them but never writes them,
    # and never proposes edits whose target is one of them.
    protected: list[str] = field(default_factory=list)
    # Optional generated note where the self-model is published inside the vault.
    report_note: str | None = None
    # Where accepted agent-drafted notes are written. Never overwrites a file.
    inbox_dir: str = "inbox"
    state_dir: Path | None = None

    encoder_kind: str = "lsa"
    encoder_dim: int = 128
    encoder_command: list[str] = field(default_factory=list)
    max_features: int = 60000

    semantic_k: int = 6
    semantic_min_sim: float = 0.30

    hebbian_lr: float = 0.25
    hebbian_half_life_days: float = 30.0
    hebbian_prune_below: float = 0.04

    gap_threshold: float = 0.22
    duplicate_threshold: float = 0.93

    host: str = "127.0.0.1"
    port: int = 7317

    inject_context: bool = True
    inject_min_score: float = 0.55
    inject_max_notes: int = 3

    agent_model: str = "claude-opus-5-5"
    agent_max_tokens: int = 16000
    agent_max_turns: int = 8
    agent_budget_usd: float = 2.0

    sleep_interval_minutes: float = 360.0
    poll_seconds: float = 5.0
    evolution_lambda: int = 8
    seed: int = 7

    params: RecallParams = field(default_factory=RecallParams)

    @property
    def state(self) -> Path:
        return self.state_dir or (self.root / STATE_DIR_NAME)

    @property
    def db_path(self) -> Path:
        return self.state / "brain.db"

    @property
    def models_dir(self) -> Path:
        return self.state / "models"


_TABLE_KEYS = {
    "vault": {"include", "exclude", "exclude_dirs", "exclude_files", "protected", "report_note", "inbox_dir"},
    "encoder": {"kind": "encoder_kind", "dim": "encoder_dim", "command": "encoder_command",
                "max_features": "max_features"},
    "graph": {"semantic_k", "semantic_min_sim", "duplicate_threshold"},
    "hebbian": {"lr": "hebbian_lr", "half_life_days": "hebbian_half_life_days",
                "prune_below": "hebbian_prune_below"},
    "recall": {"gap_threshold"},
    "server": {"host", "port"},
    "hooks": {"inject_context", "inject_min_score", "inject_max_notes"},
    "agent": {"model": "agent_model", "max_tokens": "agent_max_tokens",
              "max_turns": "agent_max_turns", "budget_usd": "agent_budget_usd"},
    "daemon": {"sleep_interval_minutes", "poll_seconds", "evolution_lambda", "seed"},
}


def find_root(start: Path | None = None) -> Path:
    """Walk up from `start` to the nearest directory holding `myelin.toml`."""
    env = os.environ.get("MYELIN_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / CONFIG_NAME).is_file():
            return candidate
    return here


def load_config(root: Path | None = None, path: Path | None = None) -> Config:
    root = (root or find_root()).resolve()
    cfg = Config(root=root)
    cfg_path = path or root / CONFIG_NAME
    if cfg_path.is_file():
        with cfg_path.open("rb") as handle:
            raw = tomllib.load(handle)
        _apply(cfg, raw)
    state_env = os.environ.get("MYELIN_STATE_DIR")
    if state_env:
        cfg.state_dir = Path(state_env).resolve()
    return cfg


def _apply(cfg: Config, raw: dict[str, Any]) -> None:
    for table, keys in _TABLE_KEYS.items():
        section = raw.get(table) or {}
        if isinstance(keys, set):
            mapping = {k: k for k in keys}
        else:
            mapping = keys
        for key, value in section.items():
            if key not in mapping:
                raise ValueError(f"unknown setting [{table}].{key} in {CONFIG_NAME}")
            setattr(cfg, mapping[key], value)
    pinned = raw.get("params")
    if pinned:
        cfg.params = RecallParams.from_dict(pinned)
    # Excluding our own state is not optional.
    if ".myelin" not in cfg.exclude_dirs:
        cfg.exclude_dirs.append(".myelin")
