"""A small reverse-mode autodiff engine on numpy.

Why not PyTorch: the brain has to train on any machine numpy installs on, in
seconds, on graphs of a few thousand nodes. That is well inside what numpy does
comfortably, and a 2 GB dependency for it would be a bad trade. The engine is
small enough to audit and every op's gradient is verified against central finite
differences in `tests/test_autograd.py`.

Only the ops the models need exist. Each op returns a new `Tensor` holding a
closure that accumulates gradients into its parents.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

DTYPE = np.float64


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward")

    def __init__(self, data, requires_grad: bool = False,
                 _parents: tuple["Tensor", ...] = (), _backward: Callable[[], None] | None = None):
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad or any(p.requires_grad for p in _parents)
        self._parents = _parents
        self._backward = _backward

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def _acc(self, g: np.ndarray) -> None:
        if not self.requires_grad:
            return
        g = _unbroadcast(g, self.data.shape)
        self.grad = g if self.grad is None else self.grad + g

    def backward(self, grad: np.ndarray | None = None) -> None:
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:  # iterative topological sort; deep graphs must not recurse
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if parent.requires_grad and id(parent) not in seen:
                    stack.append((parent, False))
        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=DTYPE)
        for node in reversed(order):
            if node._backward is not None:
                node._backward()

    # Operator sugar
    def __add__(self, other): return add(self, _t(other))
    def __radd__(self, other): return add(_t(other), self)
    def __sub__(self, other): return add(self, neg(_t(other)))
    def __rsub__(self, other): return add(_t(other), neg(self))
    def __mul__(self, other): return mul(self, _t(other))
    def __rmul__(self, other): return mul(_t(other), self)
    def __matmul__(self, other): return matmul(self, other)
    def __neg__(self): return neg(self)


def _t(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _unbroadcast(g: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g


def add(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data + b.data, _parents=(a, b))
    def bw():
        a._acc(out.grad)
        b._acc(out.grad)
    out._backward = bw
    return out


def neg(a: Tensor) -> Tensor:
    out = Tensor(-a.data, _parents=(a,))
    out._backward = lambda: a._acc(-out.grad)
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data * b.data, _parents=(a, b))
    def bw():
        a._acc(out.grad * b.data)
        b._acc(out.grad * a.data)
    out._backward = bw
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data @ b.data, _parents=(a, b))
    def bw():
        a._acc(out.grad @ b.data.T)
        b._acc(a.data.T @ out.grad)
    out._backward = bw
    return out


def relu(a: Tensor) -> Tensor:
    mask = a.data > 0
    out = Tensor(a.data * mask, _parents=(a,))
    out._backward = lambda: a._acc(out.grad * mask)
    return out


def tanh(a: Tensor) -> Tensor:
    y = np.tanh(a.data)
    out = Tensor(y, _parents=(a,))
    out._backward = lambda: a._acc(out.grad * (1.0 - y * y))
    return out


def sigmoid(a: Tensor) -> Tensor:
    y = _sigmoid(a.data)
    out = Tensor(y, _parents=(a,))
    out._backward = lambda: a._acc(out.grad * y * (1.0 - y))
    return out


def absolute(a: Tensor) -> Tensor:
    sign = np.sign(a.data)
    out = Tensor(np.abs(a.data), _parents=(a,))
    out._backward = lambda: a._acc(out.grad * sign)
    return out


def sum_(a: Tensor, axis: int | None = None, keepdims: bool = False) -> Tensor:
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims), _parents=(a,))
    def bw():
        g = out.grad
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        a._acc(np.broadcast_to(g, a.data.shape))
    out._backward = bw
    return out


def mean(a: Tensor) -> Tensor:
    return mul(sum_(a), Tensor(1.0 / a.data.size))


def concat(parts: list[Tensor], axis: int = 1) -> Tensor:
    out = Tensor(np.concatenate([p.data for p in parts], axis=axis), _parents=tuple(parts))
    def bw():
        offsets = np.cumsum([0] + [p.data.shape[axis] for p in parts])
        for p, lo, hi in zip(parts, offsets[:-1], offsets[1:]):
            idx = [slice(None)] * out.grad.ndim
            idx[axis] = slice(lo, hi)
            p._acc(out.grad[tuple(idx)])
    out._backward = bw
    return out


def take_rows(a: Tensor, idx: np.ndarray) -> Tensor:
    idx = np.asarray(idx, dtype=np.int64)
    out = Tensor(a.data[idx], _parents=(a,))
    def bw():
        g = np.zeros_like(a.data)
        np.add.at(g, idx, out.grad)
        a._acc(g)
    out._backward = bw
    return out


def dropout(a: Tensor, rate: float, rng: np.random.Generator, training: bool) -> Tensor:
    if not training or rate <= 0:
        return a
    mask = (rng.random(a.data.shape) >= rate) / (1.0 - rate)
    return mul(a, Tensor(mask))


class SparseAdj:
    """A fixed sparse matrix in COO form, used as the left operand of spmm."""

    def __init__(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n: int):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.vals = np.asarray(vals, dtype=DTYPE)
        self.n = n

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n, x.shape[1]), dtype=DTYPE)
        np.add.at(out, self.rows, self.vals[:, None] * x[self.cols])
        return out

    def apply_t(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n, x.shape[1]), dtype=DTYPE)
        np.add.at(out, self.cols, self.vals[:, None] * x[self.rows])
        return out

    @classmethod
    def normalized(cls, n: int, pairs: Iterable[tuple[int, int]], weights: Iterable[float] | None = None) -> "SparseAdj":
        """D^-1/2 (A + I) D^-1/2 over an undirected edge list (Kipf & Welling 2017)."""
        pairs = list(pairs)
        w = list(weights) if weights is not None else [1.0] * len(pairs)
        rows = [i for i, _ in pairs] + [j for _, j in pairs] + list(range(n))
        cols = [j for _, j in pairs] + [i for i, _ in pairs] + list(range(n))
        vals = np.array(w + w + [1.0] * n, dtype=DTYPE)
        rows_a, cols_a = np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)
        deg = np.zeros(n, dtype=DTYPE)
        np.add.at(deg, rows_a, vals)
        inv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        return cls(rows_a, cols_a, vals * inv[rows_a] * inv[cols_a], n)


def spmm(adj: SparseAdj, x: Tensor) -> Tensor:
    out = Tensor(adj.apply(x.data), _parents=(x,))
    out._backward = lambda: x._acc(adj.apply_t(out.grad))
    return out


def bce_with_logits(logits: Tensor, targets: np.ndarray, weights: np.ndarray | None = None) -> Tensor:
    """Mean binary cross-entropy, computed stably from logits in one fused op."""
    z = logits.data
    y = np.asarray(targets, dtype=DTYPE).reshape(z.shape)
    w = np.ones_like(z) if weights is None else np.asarray(weights, dtype=DTYPE).reshape(z.shape)
    loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    out = Tensor((w * loss).sum() / w.sum(), _parents=(logits,))
    out._backward = lambda: logits._acc(out.grad * w * (_sigmoid(z) - y) / w.sum())
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return _sigmoid(np.asarray(x, dtype=DTYPE))
