"""End-to-end optimization tests using scipy.optimize.

Each test loads a problem via the optigtest class, runs a real optimizer,
and checks that the solution is close to the known optimum (or Pareto-optimal).
"""



import numpy as np
import numpy.typing as npt
import pytest
from typing import Any
from scipy.optimize import minimize, OptimizeResult

# from pyOptiGTest.pyOptiGTest import optigtest
from pyOptiGTest import optigtest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rosenbrock_pb() -> optigtest:
    """Unconstrained Rosenbrock problem in 5 dimensions."""
    return optigtest("Rosenbrock", dim=5)


@pytest.fixture
def rosenbrock_disk_pb() -> optigtest:
    """Constrained RosenbrockDisk problem (2-D, 1 constraint)."""
    return optigtest("RosenbrockDisk")


@pytest.fixture
def binh_korn_pb() -> optigtest:
    """Multi-objective BinhKorn problem (2-D, 2 objectives, 2 constraints)."""
    return optigtest("BinhKorn")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scalar_obj(
    pb: optigtest,
    X_2d: npt.NDArray[np.floating],
    grad: bool,
) -> float | tuple[float, npt.NDArray[np.floating]]:
    """Evaluate single-objective with optional gradient."""
    if grad:
        f, g = pb.evalObj(X_2d, grad=True)
        return float(f[0]), g[0].astype(float)
    return float(pb.evalObj(X_2d)[0])


def _cons_value(pb: optigtest, X_2d: npt.NDArray[np.floating], idx: int) -> float:
    """Return scalar constraint value for constraint *idx*."""
    c = pb.evalCons(X_2d, num=[idx])
    return float(c[0])


# ---------------------------------------------------------------------------
# Tests — Unconstrained
# ---------------------------------------------------------------------------

class TestUnconstrainedOptimization:
    """Minimize Rosenbrock with L-BFGS-B and check the result."""

    def test_rosenbrock_converges(self, rosenbrock_pb: optigtest) -> None:
        pb = rosenbrock_pb

        def objective(x):
            return _scalar_obj(pb, x.reshape(1, -1), grad=True)

        x0 = np.zeros(5)
        bounds = list(zip(pb.getXmin(), pb.getXmax()))
        result = minimize(objective, x0, method="L-BFGS-B", jac=True,
                          bounds=bounds)

        assert result.success or result.fun < 1e-6
        # Rosenbrock global minimum is 0 (auto-discovered DB stores nan)
        assert result.fun == pytest.approx(0.0, abs=1e-4)

    def test_rosenbrock_minimizer(self, rosenbrock_pb: optigtest) -> None:
        pb = rosenbrock_pb

        def objective(x):
            return _scalar_obj(pb, x.reshape(1, -1), grad=True)

        x0 = np.zeros(5)
        bounds = list(zip(pb.getXmin(), pb.getXmax()))
        result = minimize(objective, x0, method="L-BFGS-B", jac=True,
                          bounds=bounds)

        expected_x = np.ones(5)  # Rosenbrock minimum at [1, 1, ..., 1]
        np.testing.assert_allclose(result.x, expected_x, atol=1e-3)


# ---------------------------------------------------------------------------
# Tests — Constrained
# ---------------------------------------------------------------------------

class TestConstrainedOptimization:
    """Minimize RosenbrockDisk with SLSQP and verify feasibility + optimality."""

    def test_rosenbrock_disk_converges(self, rosenbrock_disk_pb: optigtest) -> None:
        pb = rosenbrock_disk_pb

        def objective(x):
            return _scalar_obj(pb, x.reshape(1, -1), grad=True)

        # constraint is g(x) <= 0; scipy 'ineq' expects h(x) >= 0 → h = -g
        def constraint(x):
            return -_cons_value(pb, x.reshape(1, -1), 0)

        x0 = np.array([0.5, 0.5])
        bounds = list(zip(pb.getXmin(), pb.getXmax()))
        result = minimize(objective, x0, method="SLSQP", jac=True,
                          bounds=bounds,
                          constraints={"type": "ineq", "fun": constraint})

        assert result.success or result.fun < 1e-4
        assert result.fun == pytest.approx(pb.getGlobZmin(), abs=1e-3)

    def test_rosenbrock_disk_feasible(self, rosenbrock_disk_pb: optigtest) -> None:
        pb = rosenbrock_disk_pb

        def objective(x):
            return _scalar_obj(pb, x.reshape(1, -1), grad=True)

        def constraint(x):
            return -_cons_value(pb, x.reshape(1, -1), 0)

        x0 = np.array([0.5, 0.5])
        bounds = list(zip(pb.getXmin(), pb.getXmax()))
        result = minimize(objective, x0, method="SLSQP", jac=True,
                          bounds=bounds,
                          constraints={"type": "ineq", "fun": constraint})

        # Check feasibility via the class helper
        feasible = pb.checkCons(result.x.reshape(1, -1))
        assert feasible[0], "Optimal point should be feasible"

    def test_rosenbrock_disk_minimizer(self, rosenbrock_disk_pb: optigtest) -> None:
        pb = rosenbrock_disk_pb

        def objective(x):
            return _scalar_obj(pb, x.reshape(1, -1), grad=True)

        def constraint(x):
            return -_cons_value(pb, x.reshape(1, -1), 0)

        x0 = np.array([0.5, 0.5])
        bounds = list(zip(pb.getXmin(), pb.getXmax()))
        result = minimize(objective, x0, method="SLSQP", jac=True,
                          bounds=bounds,
                          constraints={"type": "ineq", "fun": constraint})

        expected_x = pb.getGlobXmin()
        np.testing.assert_allclose(result.x, expected_x, atol=1e-2)


# ---------------------------------------------------------------------------
# Tests — Multi-Objective (weighted-sum scalarization)
# ---------------------------------------------------------------------------

class TestMultiObjectiveOptimization:
    """Approximate Pareto front of BinhKorn via weighted-sum scalarization."""

    @staticmethod
    def _solve_weighted(pb: optigtest, w1: float) -> OptimizeResult:
        """Solve BinhKorn for a given weight w1 on f1 (w2 = 1 - w1)."""
        w2 = 1.0 - w1

        def objective(x):
            X = x.reshape(1, -1)
            f1, g1 = pb.evalObj(X, grad=True, num=[0])
            f2, g2 = pb.evalObj(X, grad=True, num=[1])
            f = w1 * float(f1[0]) + w2 * float(f2[0])
            g = w1 * g1[0] + w2 * g2[0]
            return f, g.astype(float)

        # cons1 <= 0 → scipy ineq: -cons1 >= 0
        def c1(x):
            return -_cons_value(pb, x.reshape(1, -1), 0)

        # cons2 >= 0 → scipy ineq: cons2 >= 0
        def c2(x):
            return _cons_value(pb, x.reshape(1, -1), 1)

        bounds = list(zip(pb.getXmin(), pb.getXmax()))
        constraints = [
            {"type": "ineq", "fun": c1},
            {"type": "ineq", "fun": c2},
        ]
        x0 = np.array([2.5, 1.5])
        return minimize(objective, x0, method="SLSQP", jac=True,
                        bounds=bounds, constraints=constraints)

    def test_binh_korn_pareto_feasible(self, binh_korn_pb: optigtest) -> None:
        """All Pareto-optimal points should satisfy constraints."""
        pb = binh_korn_pb
        for w1 in np.linspace(0.1, 0.9, 5):
            res = self._solve_weighted(pb, w1)
            if res.success:
                feasible = pb.checkCons(res.x.reshape(1, -1))
                assert feasible[0], f"Point for w1={w1} should be feasible"

    def test_binh_korn_pareto_diversity(self, binh_korn_pb: optigtest) -> None:
        """Different weights should yield distinct trade-off points."""
        pb = binh_korn_pb
        points = []
        for w1 in np.linspace(0.05, 0.95, 10):
            res = self._solve_weighted(pb, w1)
            if res.success:
                X = res.x.reshape(1, -1)
                f1 = float(pb.evalObj(X, num=[0])[0])
                f2 = float(pb.evalObj(X, num=[1])[0])
                points.append((f1, f2))

        points = np.array(points)
        assert len(points) >= 5, "Should find at least 5 Pareto points"
        # Check there is actual spread in both objectives
        assert np.ptp(points[:, 0]) > 1.0, "f1 should vary across Pareto front"
        assert np.ptp(points[:, 1]) > 1.0, "f2 should vary across Pareto front"

    def test_binh_korn_extreme_weights(self, binh_korn_pb: optigtest) -> None:
        """Weight near 1.0 should favour f1; weight near 0.0 should favour f2."""
        pb = binh_korn_pb
        res_f1 = self._solve_weighted(pb, 0.99)
        res_f2 = self._solve_weighted(pb, 0.01)
        assert res_f1.success and res_f2.success

        X1 = res_f1.x.reshape(1, -1)
        X2 = res_f2.x.reshape(1, -1)
        f1_at_w1 = float(pb.evalObj(X1, num=[0])[0])
        f1_at_w2 = float(pb.evalObj(X2, num=[0])[0])
        f2_at_w1 = float(pb.evalObj(X1, num=[1])[0])
        f2_at_w2 = float(pb.evalObj(X2, num=[1])[0])

        # Favouring f1 should give a smaller f1
        assert f1_at_w1 <= f1_at_w2 + 1e-6
        # Favouring f2 should give a smaller f2
        assert f2_at_w2 <= f2_at_w1 + 1e-6
