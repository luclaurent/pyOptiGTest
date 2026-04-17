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


class FunObjZDT22(TestFunction):
    def evaluate(self, X, grad=False):
        """
        ZDT2 2nd objective function.
        f2(x) = g(x) * [1 - (x1/g(x))^2]
        where g(x) = 1 + 9/(n-1) * sum(x2..xn)
        Design space: 0 <= xi <= 1, dim=30
        """

        n = X.shape[1]
        x1 = X[:, 0]
        g = 1 + 9 / (n - 1) * np.sum(X[:, 1:], axis=1)
        p = g * (1 - (x1 / g)**2)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = -2 * x1 / g
            dg = 9 / (n - 1)
            for j in range(1, n):
                dp[:, j] = dg * (1 + (x1 / g)**2)
            return p, dp
        return p


funObjZDT22 = FunObjZDT22()
