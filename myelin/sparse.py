"""Minimal CSR sparse matrices on numpy alone.

scipy would do this better; it is not a dependency because the core must install
anywhere numpy does. Only the three operations the encoder and the graph need are
implemented: sparse @ dense, its transpose, and row normalisation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CSR:
    indptr: np.ndarray   # (n_rows + 1,)
    indices: np.ndarray  # (nnz,)
    data: np.ndarray     # (nnz,)
    shape: tuple[int, int]

    @classmethod
    def from_rows(cls, rows: list[dict[int, float]], n_cols: int) -> "CSR":
        indptr = np.zeros(len(rows) + 1, dtype=np.int64)
        idx: list[int] = []
        vals: list[float] = []
        for i, row in enumerate(rows):
            items = sorted(row.items())
            idx.extend(k for k, _ in items)
            vals.extend(v for _, v in items)
            indptr[i + 1] = len(idx)
        return cls(indptr, np.asarray(idx, dtype=np.int64), np.asarray(vals, dtype=np.float64),
                   (len(rows), n_cols))

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1])

    def row_ids(self) -> np.ndarray:
        return np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))

    def matmul(self, dense: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """self @ dense, processed in row chunks to bound the nnz×k temporary."""
        out = np.zeros((self.shape[0], dense.shape[1]), dtype=dense.dtype)
        for start in range(0, self.shape[0], chunk):
            stop = min(start + chunk, self.shape[0])
            lo, hi = self.indptr[start], self.indptr[stop]
            if hi == lo:
                continue
            prod = self.data[lo:hi, None].astype(dense.dtype) * dense[self.indices[lo:hi]]
            local = self.indptr[start:stop + 1] - lo
            counts = np.diff(local)
            nonempty = counts > 0
            sums = np.add.reduceat(prod, local[:-1][nonempty], axis=0)
            out[start:stop][nonempty] = sums
        return out

    def transpose(self) -> "CSR":
        rows = self.row_ids()
        order = np.lexsort((rows, self.indices))
        cols_sorted = self.indices[order]
        indptr = np.zeros(self.shape[1] + 1, dtype=np.int64)
        np.add.at(indptr, cols_sorted + 1, 1)
        indptr = np.cumsum(indptr)
        return CSR(indptr, rows[order], self.data[order], (self.shape[1], self.shape[0]))

    def l2_normalize_rows(self) -> "CSR":
        sq = np.zeros(self.shape[0])
        np.add.at(sq, self.row_ids(), self.data ** 2)
        norms = np.sqrt(sq)
        norms[norms == 0] = 1.0
        return CSR(self.indptr, self.indices, self.data / np.repeat(norms, np.diff(self.indptr)), self.shape)

    def to_dense(self) -> np.ndarray:
        out = np.zeros(self.shape)
        out[self.row_ids(), self.indices] = self.data
        return out


def randomized_svd(a: CSR, k: int, oversample: int = 10, power_iters: int = 2,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Halko–Martinsson–Tropp randomized SVD. Returns (U, s, Vt) with k components."""
    n, m = a.shape
    k = max(1, min(k, n, m))
    width = min(k + oversample, n, m)
    rng = np.random.default_rng(seed)
    at = a.transpose()
    omega = rng.standard_normal((m, width)).astype(np.float32)
    y = a.matmul(omega)
    q, _ = np.linalg.qr(y)
    for _ in range(power_iters):
        z, _ = np.linalg.qr(at.matmul(q))
        q, _ = np.linalg.qr(a.matmul(z))
    b = at.matmul(q).T.astype(np.float64)    # (width, m) = Q^T A
    ub, s, vt = np.linalg.svd(b, full_matrices=False)
    u = q @ ub
    # Sign convention: make the largest-magnitude entry of each component positive
    # so repeated fits on the same corpus produce identical vectors.
    signs = np.sign(vt[np.arange(vt.shape[0]), np.abs(vt).argmax(axis=1)])
    signs[signs == 0] = 1
    return u[:, :k] * signs[:k], s[:k], vt[:k] * signs[:k, None]
