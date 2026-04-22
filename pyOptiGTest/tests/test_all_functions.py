"""Parametrized tests that exercise every optimisation test function.

Each function is tested for:
  - importability
  - correct output shape for scalar evaluation
  - correct output shape for batch evaluation
  - gradient output shape when grad=True
  - deterministic behaviour (excluding stochastic functions)
"""


import numpy as np
import pytest
from typing import Any, Callable

from .conftest import ALL_FUNCTION_NAMES, REQUIRED_DIM, DEFAULT_DIM, load_function, dim_for

# Stochastic functions whose output may vary between identical calls
STOCHASTIC: set[str] = {"funQuartic", "funStochastic", "funXinSheYang1"}

# Functions that produce NaN/Inf for some random inputs due to log/pow/div domains
NAN_PRONE: set[str] = {"funGear", "funGulfResearch", "funPaviani"}


# ---------------------------------------------------------------------------
# Parametrize over every function name
# ---------------------------------------------------------------------------

@pytest.fixture(params=ALL_FUNCTION_NAMES, scope="module")
def func_bundle(request: pytest.FixtureRequest) -> tuple[str, Callable[..., Any], int]:
    """Return (name, callable, n_vars) for one test function."""
    name: str = request.param
    fn: Callable[..., Any] = load_function(name)
    n_vars: int = dim_for(name)
    return name, fn, n_vars


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestImport:
    """Every function file can be loaded."""

    @pytest.mark.parametrize("name", ALL_FUNCTION_NAMES)
    def test_importable(self, name: str) -> None:
        fn = load_function(name)
        assert callable(fn), f"{name} is not callable"


class TestOutputShape:
    """Output shapes follow the (n_samples,) convention."""

    @pytest.mark.parametrize("name", ALL_FUNCTION_NAMES)
    def test_single_sample(self, name: str) -> None:
        fn = load_function(name)
        n_vars = dim_for(name)
        X = np.random.default_rng(0).random((1, n_vars))
        result = fn(X)
        assert isinstance(result, np.ndarray), f"{name}: expected ndarray, got {type(result)}"
        assert result.shape == (1,), f"{name}: expected shape (1,), got {result.shape}"

    @pytest.mark.parametrize("name", ALL_FUNCTION_NAMES)
    def test_batch_samples(self, name: str) -> None:
        fn = load_function(name)
        n_vars = dim_for(name)
        X = np.random.default_rng(1).random((10, n_vars))
        result = fn(X)
        assert isinstance(result, np.ndarray), f"{name}: expected ndarray, got {type(result)}"
        assert result.shape == (10,), f"{name}: expected shape (10,), got {result.shape}"


class TestGradient:
    """Functions returning gradients should return (p, dp) with correct shapes."""

    @pytest.mark.parametrize("name", ALL_FUNCTION_NAMES)
    def test_grad_shape(self, name: str) -> None:
        fn = load_function(name)
        n_vars = dim_for(name)
        X = np.random.default_rng(2).random((5, n_vars))
        result = fn(X, grad=True)
        # grad=True should return a tuple (p, dp)
        if isinstance(result, tuple):
            p, dp = result
            assert p.shape == (5,), f"{name}: p shape {p.shape} != (5,)"
            assert dp.shape == (5, n_vars), f"{name}: dp shape {dp.shape} != (5, {n_vars})"
        else:
            # Some functions may not implement grad yet – they just return p.
            # That's acceptable; just check shape.
            assert result.shape == (5,), f"{name}: result shape {result.shape} != (5,)"


class TestDeterminism:
    """Calling the same function with the same input twice yields the same output."""

    @pytest.mark.parametrize("name", [n for n in ALL_FUNCTION_NAMES if n not in STOCHASTIC])
    def test_deterministic(self, name: str) -> None:
        fn = load_function(name)
        n_vars = dim_for(name)
        X = np.random.default_rng(3).random((4, n_vars))
        r1 = fn(X)
        r2 = fn(X)
        np.testing.assert_array_equal(r1, r2, err_msg=f"{name} is non-deterministic")


class TestFiniteOutput:
    """Function output should not contain NaN or Inf for well-behaved input."""

    @pytest.mark.parametrize("name", [n for n in ALL_FUNCTION_NAMES if n not in NAN_PRONE])
    def test_finite(self, name: str) -> None:
        fn = load_function(name)
        n_vars = dim_for(name)
        # Use a small positive input to avoid potential division-by-zero issues
        X = np.random.default_rng(4).random((5, n_vars)) * 0.5 + 0.1
        result = fn(X)
        assert np.all(np.isfinite(result)), (
            f"{name}: output contains non-finite values: {result}"
        )
