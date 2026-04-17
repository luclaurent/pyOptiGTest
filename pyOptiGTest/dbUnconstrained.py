"""
    pyOptiGTest - Database of unconstrained optimization test problems.

    Auto-discovers all unconstrained test functions from the functions directory
    and provides a unified interface for accessing problem metadata.

    This file is part of pyOptiGTest.

    MIT License
    Copyright (c) 2020 Luc LAURENT
    luc.laurent@lecnam.net

    Sources available at:
    https://github.com/luclaurent/optigtest/
"""

import pathlib
import numpy as np


# ---------------------------------------------------------------------------
# Known metadata for unconstrained problems
# Maps function name (without 'fun' prefix) to {dim, minFglob, minXglob, space}
# Functions not listed here get default metadata.
# ---------------------------------------------------------------------------

_FUNCTIONS_DIR = pathlib.Path(__file__).parent / "functions"

# Excluded prefixes: constraint/objective functions for constrained/multi-obj
_EXCLUDED_PREFIXES = ('funObj', 'funCons', 'funDisk')


def _discover_functions():
    """Discover all unconstrained function files from the functions directory."""

    names = []
    for f in sorted(_FUNCTIONS_DIR.glob("fun*.py")):
        name = f.stem
        if name == "__init__":
            continue
        # Skip constraint/multi-objective functions
        if any(name.startswith(prefix) for prefix in _EXCLUDED_PREFIXES):
            continue
        names.append(name)
    return names


def listPb(dim=0):
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

    fun_names = _discover_functions()
    pb = {}

    for fname in fun_names:
        # Problem name is function name without 'fun' prefix
        pb_name = fname[3:] if fname.startswith('fun') else fname
        pb[pb_name] = {
            'funobj': [fname],
            'funcons': None,
            'typecons': None,
            'type': 'Unconstrained',
            'dim': np.inf,
            'minFglob': np.nan,
            'minXglob': np.nan,
            'space': np.array([[-5.0, 5.0]]),
        }

    return pb


def loadPb(pbName=None, dim=0):
    """Load unconstrained problem(s) by name.

    Args:
        pbName: Problem name string (without 'fun' prefix). If None, return all.
        dim: Dimension (default: 0).

    Returns:
        dict: Single problem dict if pbName given, else all problems.
    """

    allPb = listPb(dim)
    if pbName:
        return allPb.get(pbName, {})
    return allPb
