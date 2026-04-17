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


class FunCons2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Constraint function 2 for RosenbrockCubicLine problem.
        g(x) = x1 + x2 - 2 <= 0
        """

        xxx = X[:, 0]
        yyy = X[:, 1]

        p = xxx + yyy - 2

        if grad:
            dp = np.ones_like(X)
            return p, dp
        return p


funCons2 = FunCons2()
