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


class FunConsTestFun43(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Test Function 4 constraint 3.
        g3(x) = x2 + 9*x1 >= 1  (corrected from MATLAB: x2+9x1>=1)
        (returned value should be >= 0 for feasibility)
        """

        p = X[:, 1] + 9 * X[:, 0] - 1

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 9 * np.ones(X.shape[0])
            dp[:, 1] = np.ones(X.shape[0])
            return p, dp
        return p


funConsTestFun43 = FunConsTestFun43()
