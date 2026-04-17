"""
    pyOptiGTest - Python implementation of optimization test functions.

    This file is part of pyOptiGTest.

    MIT License
    Copyright (c) 2020 Luc LAURENT
    luc.laurent@lecnam.net

    Sources available at:
    https://github.com/luclaurent/optigtest/
"""

import numpy as np

from pyOptiGTest.base import TestFunction


class FunCons1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Constraint function 1 for RosenbrockCubicLine problem.
        g(x) = (x1-1)^3 - x2 + 1 <= 0
        """

        xxx = X[:, 0]
        yyy = X[:, 1]

        p = (xxx - 1)**3 - yyy + 1

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 3 * (xxx - 1)**2
            dp[:, 1] = -np.ones_like(yyy)
            return p, dp
        return p


funCons1 = FunCons1()
