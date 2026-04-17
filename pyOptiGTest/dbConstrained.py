"""
    pyOptiGTest - Database of constrained optimization test problems.

    This file is part of pyOptiGTest.

    MIT License
    Copyright (c) 2020 Luc LAURENT
    luc.laurent@lecnam.net

    Sources available at:
    https://github.com/luclaurent/optigtest/
"""

import json
import pathlib

import numpy as np

_JSON_FILE = pathlib.Path(__file__).parent / "dbConstrained.json"


def _convert_entry(entry):
    """Convert a JSON entry to the expected Python types (numpy arrays, etc.)."""
    dim = entry.get("dim")
    if dim == "inf":
        entry["dim"] = np.inf
    elif isinstance(dim, (int, float)):
        entry["dim"] = int(dim)
    for key in ("minFglob",):
        if entry.get(key) is None:
            entry[key] = np.nan
    for key in ("minXglob",):
        v = entry.get(key)
        if v is None:
            entry[key] = np.nan
        elif isinstance(v, list):
            entry[key] = np.asarray(v)
    if entry.get("space") is not None:
        entry["space"] = np.asarray(entry["space"])
    return entry


def listPb():
    """Return a dictionary of all constrained optimization test problems.

    Each entry maps a problem name to a dict with keys:
        - funobj: list of objective function names
        - funcons: list of constraint function names
        - typecons: list of constraint types ('<=', '<', '>=')
        - type: 'Constrained'
        - dim: number of design variables (int or np.inf)
        - minFglob: known global minimum value (float or np.nan)
        - minXglob: known global minimizer (numpy array or np.nan)
        - space: design space bounds as numpy array, shape (dim, 2)
    """
    with open(_JSON_FILE, "r") as f:
        raw = json.load(f)
    return {name: _convert_entry(data) for name, data in raw.items()}


def loadPb(pbName=None):
    """Load constrained problem(s) by name.

    Args:
        pbName: Problem name string. If None, return all problems.

    Returns:
        dict: Single problem dict if pbName given, else all problems.
    """

    allPb = listPb()
    if pbName:
        return allPb.get(pbName, {})
    return allPb