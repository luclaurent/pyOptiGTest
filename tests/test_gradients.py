"""Finite-difference gradient verification for select functions.

Checks that the analytical gradient (when available) approximately matches
a central-difference numerical gradient.
"""



import numpy as np
import numpy.typing as npt
import pytest
from typing import Callable, Any

from conftest import load_function, dim_for

# Functions to verify gradients for (must implement grad=True properly)
GRAD_FUNCTIONS: list[str] = [
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

FD_STEP: float = 1e-6
FD_RTOL: float = 1e-3
FD_ATOL: float = 1e-4


def _numerical_grad(
    fn: Callable[..., Any],
    X: npt.NDArray[np.floating],
    h: float = FD_STEP,
) -> npt.NDArray[np.floating]:
    """Central-difference numerical gradient, shape (n_samples, n_vars)."""
    n_samples, n_vars = X.shape
    grad: npt.NDArray[np.floating] = np.zeros_like(X)
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
def test_gradient_matches_fd(name: str) -> None:
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
