"""Layers, an AdamW optimizer, and weight (de)serialisation for the autograd engine."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .autograd import DTYPE, SparseAdj, Tensor, matmul, relu, spmm


class Module:
    def parameters(self) -> list[Tensor]:
        out: list[Tensor] = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                out.append(value)
            elif isinstance(value, Module):
                out.extend(value.parameters())
            elif isinstance(value, list):
                out.extend(p for m in value if isinstance(m, Module) for p in m.parameters())
        return out

    def state(self) -> dict[str, np.ndarray]:
        return {f"p{i}": p.data.copy() for i, p in enumerate(self.parameters())}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        params = self.parameters()
        if len(params) != len(state):
            raise ValueError(f"state has {len(state)} tensors, model has {len(params)}")
        for i, p in enumerate(params):
            value = np.asarray(state[f"p{i}"], dtype=DTYPE)
            if value.shape != p.data.shape:
                raise ValueError(f"shape mismatch for p{i}: {value.shape} vs {p.data.shape}")
            p.data = value.copy()


def _glorot(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


class Linear(Module):
    def __init__(self, fan_in: int, fan_out: int, rng: np.random.Generator, bias: bool = True):
        self.w = Tensor(_glorot(rng, fan_in, fan_out), requires_grad=True)
        self.b = Tensor(np.zeros((1, fan_out)), requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        y = matmul(x, self.w)
        return y + self.b if self.b is not None else y


class MLP(Module):
    def __init__(self, sizes: list[int], rng: np.random.Generator):
        self.layers = [Linear(a, b, rng) for a, b in zip(sizes, sizes[1:])]

    def __call__(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = relu(x)
        return x


class GCNLayer(Module):
    """H' = Â H W + b  (activation applied by the caller)."""

    def __init__(self, fan_in: int, fan_out: int, rng: np.random.Generator):
        self.lin = Linear(fan_in, fan_out, rng)

    def __call__(self, adj: SparseAdj, h: Tensor) -> Tensor:
        # Multiply by W first when it shrinks the width: Â(HW) == (ÂH)W.
        return spmm(adj, self.lin(h))


class AdamW:
    def __init__(self, params: list[Tensor], lr: float = 0.01, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 5e-4, clip: float = 5.0):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.wd, self.clip = weight_decay, clip
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> float:
        self.t += 1
        grads = [p.grad if p.grad is not None else np.zeros_like(p.data) for p in self.params]
        norm = float(np.sqrt(sum((g * g).sum() for g in grads)))
        scale = self.clip / norm if self.clip and norm > self.clip else 1.0
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * scale
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * p.data)
        return norm


def save_weights(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_weights(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
