"""Sparse algebra, metrics, and the autograd engine's gradients."""
import numpy as np
import pytest

from myelin.metrics import average_precision, expected_calibration_error, reciprocal_rank, roc_auc
from myelin.neural import autograd as A
from myelin.sparse import CSR, randomized_svd


def _rand_csr(rng, n=40, m=70, density=0.1):
    dense = rng.random((n, m)) * (rng.random((n, m)) < density)
    dense[3] = 0  # an empty row
    rows = [{j: v for j, v in enumerate(r) if v} for r in dense]
    return CSR.from_rows(rows, m), dense


def test_csr_ops_match_dense():
    rng = np.random.default_rng(0)
    a, dense = _rand_csr(rng)
    x = rng.normal(size=(70, 5))
    assert np.allclose(a.matmul(x), dense @ x)
    assert np.allclose(a.transpose().to_dense(), dense.T)
    assert np.allclose(a.matmul(x, chunk=7), dense @ x)


def test_randomized_svd_recovers_top_singular_values():
    rng = np.random.default_rng(1)
    a, dense = _rand_csr(rng, 60, 90, 0.3)
    _, s, _ = randomized_svd(a, 5, power_iters=4)
    exact = np.linalg.svd(dense, compute_uv=False)[:5]
    assert np.allclose(s, exact, rtol=1e-2)


def test_metrics_against_hand_values():
    assert roc_auc(np.array([0.9, 0.8, 0.3, 0.1]), np.array([1, 0, 1, 0])) == pytest.approx(0.75)
    assert roc_auc(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.5)  # ties
    assert average_precision(np.array([0.9, 0.8, 0.3]), np.array([1, 0, 1])) == pytest.approx((1 + 2 / 3) / 2)
    assert reciprocal_rank([5, 7, 9], 9) == pytest.approx(1 / 3)
    assert reciprocal_rank([5, 7, 9], 4) == 0.0
    assert expected_calibration_error(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0)


def _gradcheck(build, params, eps=1e-6, tol=1e-5):
    loss = build()
    for p in params:
        p.grad = None
    loss.backward()
    for p in params:
        analytic = p.grad.copy()
        numeric = np.zeros_like(p.data)
        for idx in np.ndindex(p.data.shape):
            old = p.data[idx]
            p.data[idx] = old + eps
            hi = float(build().data)
            p.data[idx] = old - eps
            lo = float(build().data)
            p.data[idx] = old
            numeric[idx] = (hi - lo) / (2 * eps)
        assert np.allclose(analytic, numeric, atol=tol), (analytic, numeric)


def test_autograd_gradients_match_finite_differences():
    rng = np.random.default_rng(2)
    W = A.Tensor(rng.normal(size=(4, 3)), requires_grad=True)
    b = A.Tensor(rng.normal(size=(1, 3)), requires_grad=True)
    V = A.Tensor(rng.normal(size=(6, 2)), requires_grad=True)
    X = rng.normal(size=(5, 4))
    adj = A.SparseAdj.normalized(5, [(0, 1), (1, 2), (3, 4), (0, 4)])
    idx = np.array([0, 2, 2, 4])

    def build():
        h = A.spmm(adj, A.tanh(A.Tensor(X) @ W + b))           # matmul, broadcast add, tanh, spmm
        r = A.relu(h) * A.sigmoid(h) - A.absolute(h)          # relu, sigmoid, abs, mul, sub
        c = A.concat([A.take_rows(r, idx), A.take_rows(h, idx)], axis=1)
        z = c @ V
        return A.bce_with_logits(A.sum_(z, axis=1, keepdims=True), np.array([1, 0, 1, 0])) + A.mean(z)

    _gradcheck(build, [W, b, V])


def test_bce_matches_closed_form():
    z = A.Tensor(np.array([[2.0], [-1.0]]), requires_grad=True)
    loss = A.bce_with_logits(z, np.array([1, 0]))
    expected = np.mean([np.log1p(np.exp(-2.0)), np.log1p(np.exp(-1.0))])
    assert float(loss.data) == pytest.approx(expected)
