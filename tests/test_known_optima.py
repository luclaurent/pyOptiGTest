"""Verify known global optima for select functions."""



import numpy as np
import numpy.typing as npt
import pytest

from conftest import KNOWN_OPTIMA, load_function


@pytest.mark.parametrize(
    "name, x_opt, f_opt, tol",
    KNOWN_OPTIMA,
    ids=[t[0] for t in KNOWN_OPTIMA],
)
def test_known_optimum(name: str, x_opt: npt.NDArray[np.floating], f_opt: float, tol: float) -> None:
    """Evaluate the function at its known global minimum and check the value."""
    fn = load_function(name)
    result = fn(x_opt)
    assert result.shape == (x_opt.shape[0],), (
        f"{name}: shape mismatch {result.shape} vs ({x_opt.shape[0]},)"
    )
    np.testing.assert_allclose(
        result[0], f_opt, atol=tol,
        err_msg=f"{name}: f({x_opt}) = {result[0]}, expected {f_opt}",
    )
