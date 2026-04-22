"""Shared fixtures and helpers for pyOptiGTest tests."""



import importlib
import importlib.util
import json
import pathlib
from typing import Any, Callable

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# CLI option for picture generation
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--gen-pictures",
        action="store_true",
        default=False,
        help="Actually generate wiki pictures (slow).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--gen-pictures"):
        return
    skip = pytest.mark.skip(reason="need --gen-pictures option to run")
    for item in items:
        if "genpictures" in item.keywords:
            item.add_marker(skip)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent.parent
FUNCTIONS_DIR: pathlib.Path = ROOT / "pyOptiGTest" / "functions"
_JSON_FILE: pathlib.Path = pathlib.Path(__file__).resolve().parent / "conftest.json"

# ---------------------------------------------------------------------------
# Load test data from JSON
# ---------------------------------------------------------------------------

with open(_JSON_FILE, "r") as _f:
    _CONFTEST_DATA: dict[str, Any] = json.load(_f)

REQUIRED_DIM: dict[str, int] = _CONFTEST_DATA["required_dim"]
DEFAULT_DIM: int = _CONFTEST_DATA["default_dim"]

KNOWN_OPTIMA: list[tuple[str, np.ndarray, float, float]] = [
    (
        entry["function"],
        np.asarray(entry["x_optimal"]),
        entry["f_optimal"],
        entry["tolerance"],
    )
    for entry in _CONFTEST_DATA["known_optima"]
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_all_function_names() -> list[str]:
    """Return sorted list of all function names from the functions directory."""
    names = []
    for f in sorted(FUNCTIONS_DIR.glob("fun*.py")):
        name = f.stem
        if name != "__init__":
            names.append(name)
    return names


def load_function(name: str) -> Callable[..., Any]:
    """Import a single function by name from its file."""
    path = FUNCTIONS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def dim_for(name: str) -> int:
    """Return the number of dimensions to use when testing *name*."""
    return REQUIRED_DIM.get(name, DEFAULT_DIM)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_FUNCTION_NAMES: list[str] = get_all_function_names()


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def all_function_names() -> list[str]:
    """List of every function name in the package."""
    return ALL_FUNCTION_NAMES
