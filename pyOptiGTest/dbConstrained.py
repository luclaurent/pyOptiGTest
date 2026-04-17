"""
    pyOptiGTest - Database of constrained optimization test problems.

    This file is part of pyOptiGTest.

    MIT License
    Copyright (c) 2020 Luc LAURENT
    luc.laurent@lecnam.net

    Sources available at:
    https://github.com/luclaurent/optigtest/
"""

import numpy as np


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

    pb = {
        'RosenbrockCubicLine': {
            'funobj': ['funRosenbrock'],
            'funcons': ['funCons1', 'funCons2'],
            'typecons': ['<=', '<='],
            'type': 'Constrained',
            'dim': 2,
            'minFglob': 0.0,
            'minXglob': np.array([1.0, 1.0]),
            'space': np.array([[-1.5, 1.5], [-0.5, 2.5]]),
        },
        'RosenbrockDisk': {
            'funobj': ['funRosenbrock'],
            'funcons': ['funDisk2'],
            'typecons': ['<='],
            'type': 'Constrained',
            'dim': 2,
            'minFglob': 0.0,
            'minXglob': np.array([1.0, 1.0]),
            'space': np.array([[-1.5, 1.5], [-1.5, 1.5]]),
        },
        'BirdDisk': {
            'funobj': ['funBird'],
            'funcons': ['funDisk25'],
            'typecons': ['<'],
            'type': 'Constrained',
            'dim': 2,
            'minFglob': -106.764537,
            'minXglob': np.array([-1.582142, -3.130247]),
            'space': np.array([[-10.0, 0.0], [-6.5, 0.0]]),
        },
        'Townsend': {
            'funobj': ['funTownsend'],
            'funcons': ['funConsTownsend'],
            'typecons': ['<'],
            'type': 'Constrained',
            'dim': 2,
            'minFglob': -2.0239884,
            'minXglob': np.array([2.0052938, 1.1944506]),
            'space': np.array([[-2.25, 2.5], [-2.5, 1.75]]),
        },
        'Simionescu': {
            'funobj': ['funSimionescu'],
            'funcons': ['funConsSimionescu'],
            'typecons': ['<='],
            'type': 'Constrained',
            'dim': 2,
            'minFglob': -0.072,
            'minXglob': np.array([-0.84852813, 0.84852813]),
            'space': np.array([[-1.25, 1.25], [-1.25, 1.25]]),
        },
    }
    return pb


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