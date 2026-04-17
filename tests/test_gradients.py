"""Finite-difference gradient verification for select functions.

Checks that the analytical gradient (when available) approximately matches
a central-difference numerical gradient.
"""

import numpy as np
import pytest

from conftest import load_function, dim_for

# Functions to verify gradients for (must implement grad=True properly)
GRAD_FUNCTIONS = [
    "funAckley2",
    "funRosenbrock",
    "funDejong",
    "funSphere",
    "funSumSquare",
    "funAlpine1",
    "funGriewank",
    "funRastrigin",
    "funSchwefel01",
    "funChungReynolds",
]

FD_STEP = 1e-6
FD_RTOL = 1e-3
FD_ATOL = 1e-4


def _numerical_grad(fn, X, h=FD_STEP):
    """Central-difference numerical gradient, shape (n_samples, n_vars)."""
    n_samples, n_vars = X.shape
    grad = np.zeros_like(X)
    for j in range(n_vars):
        Xp = X.copy()
        Xm = X.copy()
        Xp[:, j] += h
        Xm[:, j] -= h
        fp = fn(Xp)
        fm = fn(Xm)
        grad[:, j] = (fp - fm) / (2 * h)
    return grad


@pytest.mark.parametrize("name", GRAD_FUNCTIONS)
def test_gradient_matches_fd(name):
    """Verify analytical gradient against finite-difference approximation."""
    fn = load_function(name)
    n_vars = dim_for(name)
    rng = np.random.default_rng(99)
    X = rng.random((5, n_vars)) * 2 - 1  # [-1, 1] range

    result = fn(X, grad=True)
    if not isinstance(result, tuple):
        pytest.skip(f"{name} does not return gradient tuple")

    _, dp_analytical = result
    dp_numerical = _numerical_grad(fn, X)

    np.testing.assert_allclose(
        dp_analytical,
        dp_numerical,
        rtol=FD_RTOL,
        atol=FD_ATOL,
        err_msg=f"{name}: analytical gradient does not match finite-difference",
    )
