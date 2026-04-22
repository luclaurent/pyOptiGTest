"""
    pyOptiGTest - Base abstract class for test functions.

    This file is part of pyOptiGTest.

    MIT License
    Copyright (c) 2020 Luc LAURENT
    luc.laurent@lecnam.net

    Sources available at:
    https://github.com/luclaurent/optigtest/
"""



import importlib
import json
import pathlib
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Metadata loading helpers (lazy, cached)
# ---------------------------------------------------------------------------

_DB_DIR: pathlib.Path = pathlib.Path(__file__).parent
_DB_FILES: dict[str, pathlib.Path] = {
    "Unconstrained": _DB_DIR / "dbUnconstrained.json",
    "Constrained": _DB_DIR / "dbConstrained.json",
    "MultiObjective": _DB_DIR / "dbMultiObj.json",
}


@lru_cache(maxsize=1)
def _load_all_metadata() -> dict[str, dict[str, Any]]:
    """Load and merge all JSON DB files into a single {funName: metadata} mapping.

    Keyed by *function name* (e.g. ``"funSphere"``), not problem name.
    """
    result: dict[str, dict[str, Any]] = {}
    for db_path in _DB_FILES.values():
        if not db_path.exists():
            continue
        with open(db_path, "r") as f:
            raw: dict[str, dict[str, Any]] = json.load(f)
        for _pb_name, entry in raw.items():
            fun_names: list[str] = entry.get("funobj") or []
            for fn in fun_names:
                # Prefer entries with dim="inf" (most permissive)
                if fn not in result or result[fn].get("dim") != "inf":
                    result[fn] = entry
            # Also index constraint functions so they can be looked up
            for fn in entry.get("funcons") or []:
                if fn not in result:
                    result[fn] = entry
    return result


def _metadata_for(fun_name: str) -> dict[str, Any] | None:
    """Return the DB entry for a given function name, or *None*."""
    return _load_all_metadata().get(fun_name)


# ---------------------------------------------------------------------------
# TestFunction base class
# ---------------------------------------------------------------------------

class TestFunction(ABC):
    """Abstract base class for all optimization test functions.

    Subclasses must implement the ``evaluate`` method.
    Instances are callable: ``f(X)`` is equivalent to ``f.evaluate(X)``.

    Metadata (bounds, dimension, known optima) is lazily loaded from the
    JSON database files when first accessed.
    """

    # ------------------------------------------------------------------
    # Metadata — loaded lazily from JSON DB
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Canonical function name derived from the class name."""
        cls_name: str = type(self).__name__
        # Convert "FunSphere" → "funSphere"
        if cls_name and cls_name[0].isupper():
            return cls_name[0].lower() + cls_name[1:]
        return cls_name

    @property
    def _meta(self) -> dict[str, Any] | None:
        """Lazily fetched metadata dict from the JSON database."""
        cache_attr: str = "_meta_cache"
        if not hasattr(self, cache_attr):
            object.__setattr__(self, cache_attr, _metadata_for(self.name))
        return getattr(self, cache_attr)

    @property
    def dim(self) -> int | float:
        """Expected number of variables (``int`` or ``np.inf``)."""
        if self._meta is None:
            return np.inf
        d = self._meta.get("dim")
        if d == "inf" or d is None:
            return np.inf
        return int(d)

    @property
    def space(self) -> npt.NDArray[np.floating[Any]] | None:
        """Design-space bounds as a ``(n_vars, 2)`` array, or ``None``."""
        if self._meta is None:
            return None
        s = self._meta.get("space")
        if s is None:
            return None
        return np.atleast_2d(np.asarray(s, dtype=float))

    @property
    def min_fglob(self) -> float:
        """Known global minimum value (``float`` or ``np.nan``)."""
        if self._meta is None:
            return np.nan
        v = self._meta.get("minFglob")
        return np.nan if v is None else float(v)

    @property
    def min_xglob(self) -> npt.NDArray[np.floating[Any]] | float:
        """Known global minimizer (numpy array or ``np.nan``)."""
        if self._meta is None:
            return np.nan
        v = self._meta.get("minXglob")
        if v is None:
            return np.nan
        return np.asarray(v, dtype=float)

    # ------------------------------------------------------------------
    # Bounds property
    # ------------------------------------------------------------------

    @property
    def bounds(self) -> list[tuple[float, float]] | None:
        """Design-space bounds as a list of ``(lower, upper)`` tuples.

        Compatible with ``scipy.optimize.minimize(bounds=...)``.
        Returns ``None`` if no bounds metadata is available.
        """
        s = self.space
        if s is None:
            return None
        return [(float(row[0]), float(row[1])) for row in s]

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate(
        self, X: npt.NDArray[np.floating[Any]], grad: bool = False
    ) -> npt.NDArray[np.floating[Any]] | tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Evaluate the function at *X*.

        Parameters
        ----------
        X : numpy.ndarray
            Input array of shape ``(n_samples, n_vars)``.
        grad : bool, optional
            If ``True``, also return the gradient.

        Returns
        -------
        p : numpy.ndarray
            Function values, shape ``(n_samples,)``.
        dp : numpy.ndarray, optional
            Gradient, shape ``(n_samples, n_vars)``; only when *grad* is ``True``.
        """

    # ------------------------------------------------------------------
    # __repr__ 
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts: list[str] = [f"<{type(self).__name__}"]
        parts.append(f"dim={self.dim}")
        if self.space is not None:
            lo = self.space[:, 0]
            hi = self.space[:, 1]
            if np.all(lo == lo[0]) and np.all(hi == hi[0]):
                parts.append(f"bounds=[{lo[0]}, {hi[0]}]")
            else:
                parts.append(f"bounds=({self.space.shape[0]} rows)")
        fmin: float = self.min_fglob
        if np.isfinite(fmin):
            parts.append(f"f*={fmin}")
        return " ".join(parts) + ">"

    # ------------------------------------------------------------------
    # Factory: from_name 
    # ------------------------------------------------------------------

    @staticmethod
    def from_name(fun_name: str) -> TestFunction:
        """Instantiate a :class:`TestFunction` by its function name string.

        Parameters
        ----------
        fun_name : str
            Function name, e.g. ``"funSphere"`` or ``"funObjKornBinh1"``.

        Returns
        -------
        TestFunction
            A new instance of the corresponding subclass.

        Raises
        ------
        ModuleNotFoundError
            If no module named *fun_name* exists in the functions package.
        AttributeError
            If the module does not contain a class with the expected name.
        """
        mod = importlib.import_module(f"pyOptiGTest.functions.{fun_name}")
        # Convention: class name is CamelCase version, e.g. funSphere → FunSphere
        class_name: str = fun_name[0].upper() + fun_name[1:]
        cls: type[TestFunction] = getattr(mod, class_name)
        return cls()

    # ------------------------------------------------------------------
    # Caching / memoization
    # ------------------------------------------------------------------

    def enable_cache(self) -> None:
        """Enable result caching for repeated evaluations at identical points.

        Once enabled, calling the instance will cache results keyed by the
        byte representation of the input array. Call :meth:`clear_cache` to
        free memory.
        """
        if not hasattr(self, "_cache"):
            object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "_caching_enabled", True)

    def disable_cache(self) -> None:
        """Disable result caching (results already cached are kept)."""
        object.__setattr__(self, "_caching_enabled", False)

    def clear_cache(self) -> None:
        """Remove all cached evaluation results."""
        if hasattr(self, "_cache"):
            self._cache.clear()

    def __call_cached__(
        self, X: npt.NDArray[np.floating[Any]], grad: bool = False
    ) -> npt.NDArray[np.floating[Any]] | tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        """Internal: cached version of __call__."""
        key: tuple[bytes, tuple[int, ...], bool] = (X.tobytes(), X.shape, grad)
        if key not in self._cache:
            self._cache[key] = self.evaluate(X, grad)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Input validation, caching & __call__
    # ------------------------------------------------------------------

    def __call__(
        self, X: npt.ArrayLike, grad: bool = False
    ) -> npt.NDArray[np.floating[Any]] | tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim != 2:
            raise ValueError(
                f"Expected 2-D input (n_samples, n_vars), got shape {X.shape}"
            )
        if getattr(self, "_caching_enabled", False):
            return self.__call_cached__(X, grad)
        return self.evaluate(X, grad)
