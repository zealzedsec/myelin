"""Text encoders: the learned semantic space every other component works in.

The default is **LSA** — sublinear TF-IDF over unigrams and bigrams, factorised by
randomized SVD. It is learned from the vault itself, runs offline in seconds, and
is deterministic, which matters because the evolution engine compares runs.

It is not a neural network and is not claimed to be one. The neural components
(`neural/`) sit on top of it: the graph autoencoder takes these vectors as node
features, and the reranker takes their similarities as inputs.

`CommandEncoder` is the extension point for a stronger model: any program that
reads JSON `{"texts": [...]}` on stdin and writes `{"vectors": [[...], ...]}`
(a local sentence-transformer, an Ollama embedding model, a hosted API wrapper).
"""
from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np

from .sparse import CSR, randomized_svd
from .store import sha256_file
from .text import features


class Encoder(Protocol):
    id: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


def l2n(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class LSAEncoder:
    def __init__(self, vocab: list[str], idf: np.ndarray, components: np.ndarray,
                 singular: np.ndarray, model_id: str = "lsa:unfit"):
        self.vocab = vocab
        self.index = {t: i for i, t in enumerate(vocab)}
        self.idf = idf
        self.components = components.astype(np.float32)   # (V, k)
        self.singular = singular
        self.id = model_id
        self.dim = components.shape[1]

    @classmethod
    def fit(cls, docs: list[str], dim: int = 128, max_features: int = 60000,
            min_df: int | None = None, seed: int = 0) -> "LSAEncoder":
        tokenized = [Counter(features(d)) for d in docs]
        df: Counter[str] = Counter()
        for counts in tokenized:
            df.update(counts.keys())
        n = max(1, len(docs))
        if min_df is None:
            # A term seen in one passage cannot make two passages similar; it only
            # widens every SVD pass. Exact rare-term matches are BM25's job.
            min_df = 2 if n >= 50 else 1
        # Terms in every document carry no contrast; terms in one only are noise
        # for a corpus this small unless they are all there is.
        ceiling = n if n < 5 else max(2, int(0.6 * n))
        kept = [(t, c) for t, c in df.items() if min_df <= c <= ceiling]
        kept.sort(key=lambda tc: (-tc[1], tc[0]))
        vocab = sorted(t for t, _ in kept[:max_features])
        index = {t: i for i, t in enumerate(vocab)}
        idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in vocab])
        rows = [_weights(c, index, idf) for c in tokenized]
        matrix = CSR.from_rows(rows, len(vocab)).l2_normalize_rows()
        k = max(1, min(dim, len(docs) - 1, len(vocab) - 1)) if len(vocab) > 1 else 1
        _, s, vt = randomized_svd(matrix, k, seed=seed)
        return cls(vocab, idf, vt.T, s)

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = [_weights(Counter(features(t)), self.index, self.idf) for t in texts]
        matrix = CSR.from_rows(rows, len(self.vocab)).l2_normalize_rows()
        return l2n(matrix.matmul(self.components)).astype(np.float32)

    def term_weights(self, text: str) -> dict[str, float]:
        return {self.vocab[i]: w for i, w in _weights(Counter(features(text)), self.index, self.idf).items()}

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, vocab=np.array(self.vocab, dtype=str), idf=self.idf,
                            components=self.components, singular=self.singular)
        tmp.replace(path)
        digest = sha256_file(path)
        self.id = f"lsa:{digest[:12]}"
        return digest

    @classmethod
    def load(cls, path: Path) -> "LSAEncoder":
        data = np.load(path, allow_pickle=False)
        return cls([str(t) for t in data["vocab"]], data["idf"], data["components"], data["singular"],
                   f"lsa:{sha256_file(path)[:12]}")


def _weights(counts: Counter, index: dict[str, int], idf: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    for term, c in counts.items():
        i = index.get(term)
        if i is not None:
            out[i] = (1.0 + math.log(c)) * idf[i]
    return out


class CommandEncoder:
    """Delegate to an external embedding program. The command is operator config,
    never model output, and runs without a shell."""

    def __init__(self, command: list[str], timeout: float = 120.0):
        if not command:
            raise ValueError("encoder.command is empty")
        self.command = command
        self.timeout = timeout
        self.id = "cmd:" + json.dumps(command)[:48]
        self.dim = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            proc = subprocess.run(self.command, input=json.dumps({"texts": batch}), text=True,
                                  capture_output=True, timeout=self.timeout, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"encoder command failed ({proc.returncode}): {proc.stderr[:300]}")
            vectors = np.asarray(json.loads(proc.stdout)["vectors"], dtype=np.float32)
            if vectors.shape[0] != len(batch):
                raise RuntimeError("encoder command returned the wrong number of vectors")
            out.append(vectors)
        result = l2n(np.vstack(out)) if out else np.zeros((0, self.dim), dtype=np.float32)
        self.dim = result.shape[1] if result.size else self.dim
        return result
