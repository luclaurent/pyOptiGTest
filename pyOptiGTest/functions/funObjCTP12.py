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


class FunObjCTP12(TestFunction):
    def evaluate(self, X, grad=False):
        """
        CTP1 2nd objective function.
        f2(x) = g(x2) * exp(-f1/g(x2))
        where g(x2) = 1 + x2
        Design space: 0 <= x1 <= 1, 0 <= x2 <= 1
        """

        x1 = X[:, 0]
        x2 = X[:, 1]
        g = 1 + x2
        p = g * np.exp(-x1 / g)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = -np.exp(-x1 / g)
            dp[:, 1] = np.exp(-x1 / g) * (1 + x1 / g)
            return p, dp
        return p


funObjCTP12 = FunObjCTP12()
