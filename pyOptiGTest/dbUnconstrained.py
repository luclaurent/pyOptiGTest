"""
    pyOptiGTest - Database of unconstrained optimization test problems.

    Loads all unconstrained test function metadata from a JSON file
    and provides a unified interface for accessing problem metadata.

    This file is part of pyOptiGTest.

    MIT License
    Copyright (c) 2020 Luc LAURENT
    luc.laurent@lecnam.net

    Sources available at:
    https://github.com/luclaurent/optigtest/
"""



import json
import pathlib
from typing import Any

import numpy as np

_JSON_FILE: pathlib.Path = pathlib.Path(__file__).parent / "dbUnconstrained.json"


def _convert_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON entry to the expected Python types (numpy arrays, etc.)."""
    dim: Any = entry.get("dim")
    if dim == "inf":
        entry["dim"] = np.inf
    elif isinstance(dim, (int, float)):
        entry["dim"] = int(dim)
    for key in ("minFglob",):
        if entry.get(key) is None:
            entry[key] = np.nan
    for key in ("minXglob",):
        v: Any = entry.get(key)
        if v is None:
            entry[key] = np.nan
        elif isinstance(v, list):
            entry[key] = np.asarray(v)
    if entry.get("space") is not None:
        entry["space"] = np.asarray(entry["space"])
    return entry


def listPb(dim: int = 0) -> dict[str, dict[str, Any]]:
    """Return a dictionary of all unconstrained optimization test problems.

    Each entry maps a problem name (without 'fun' prefix) to a dict with keys:
        - funobj: list with the single objective function name
        - funcons: None (unconstrained)
        - typecons: None
        - type: 'Unconstrained'
        - dim: number of design variables (int or np.inf)
        - minFglob: known global minimum value (float or np.nan)
        - minXglob: known global minimizer (numpy array or np.nan)
        - space: design space bounds as numpy array

    Args:
        dim: default dimension to use (default: 0)
    """
    with open(_JSON_FILE, "r") as f:
        raw: dict[str, dict[str, Any]] = json.load(f)
    return {name: _convert_entry(data) for name, data in raw.items()}


def loadPb(pbName: str | None = None, dim: int = 0) -> dict[str, Any]:
    """Load unconstrained problem(s) by name.

    Args:
        pbName: Problem name string (without 'fun' prefix). If None, return all.
        dim: Dimension (default: 0).

    Returns:
        dict: Single problem dict if pbName given, else all problems.
    """

    allPb: dict[str, dict[str, Any]] = listPb(dim)
    if pbName:
        return allPb.get(pbName, {})
    return allPb
