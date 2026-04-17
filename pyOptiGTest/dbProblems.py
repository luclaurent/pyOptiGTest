"""
    pyOptiGTest - Database of all optimization test problems.

    Aggregates unconstrained, constrained, and multi-objective problems
    into a single unified interface.

    This file is part of pyOptiGTest.

    MIT License
    Copyright (c) 2020 Luc LAURENT
    luc.laurent@lecnam.net

    Sources available at:
    https://github.com/luclaurent/optigtest/
"""

from . import dbUnconstrained
from . import dbConstrained
from . import dbMultiObj

import numpy as np


def loadPb(pbName=None, dim=0):
    """Load problem(s) by name from all categories.

    Args:
        pbName: Problem name string. If None, return all problems.
        dim: Dimension for unconstrained problems (default: 0).

    Returns:
        dict: Single problem dict if pbName given, else all problems.
    """

    allPb = listPb(dim)
    if pbName:
        return allPb.get(pbName, {})
    return allPb


def listPb(dim=0):
    """Return a dictionary of all optimization test problems.

    Merges unconstrained, constrained, and multi-objective problems.

    Args:
        dim: Dimension for unconstrained problems (default: 0).

    Returns:
        dict: All problems keyed by name.
    """

    pb = {}
    pb.update(dbUnconstrained.listPb(dim))
    pb.update(dbConstrained.listPb())
    pb.update(dbMultiObj.listPb())
    return pb


def listPbByType(pbType=None, dim=0):
    """Return problems filtered by type.

    Args:
        pbType: One of 'Unconstrained', 'Constrained', 'MultiObjective', or None for all.
        dim: Dimension for unconstrained problems.

    Returns:
        dict: Filtered problems.
    """

    if pbType == 'Unconstrained':
        return dbUnconstrained.listPb(dim)
    elif pbType == 'Constrained':
        return dbConstrained.listPb()
    elif pbType == 'MultiObjective':
        return dbMultiObj.listPb()
    else:
        return listPb(dim)



