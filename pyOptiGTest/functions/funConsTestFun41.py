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


class FunConsTestFun41(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Test Function 4 constraint 1.
        g1(x) = x2 + 9*x1 >= 6
        rewritten as: g1(x) = 6 - x2 - 9*x1 <= 0
        (returned value should be >= 0 for feasibility)
        """

        p = X[:, 1] + 9 * X[:, 0] - 6

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 9 * np.ones(X.shape[0])
            dp[:, 1] = np.ones(X.shape[0])
            return p, dp
        return p


funConsTestFun41 = FunConsTestFun41()
