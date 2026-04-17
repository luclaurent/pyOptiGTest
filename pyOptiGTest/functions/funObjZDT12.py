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


class FunObjZDT12(TestFunction):
    def evaluate(self, X, grad=False):
        """
        ZDT1 2nd objective function.
        f2(x) = g(x) * [1 - sqrt(x1/g(x))]
        where g(x) = 1 + 9/(n-1) * sum(x2..xn)
        Design space: 0 <= xi <= 1, dim=30
        """

        n = X.shape[1]
        x1 = X[:, 0]
        g = 1 + 9 / (n - 1) * np.sum(X[:, 1:], axis=1)
        p = g * (1 - np.sqrt(x1 / g))

        if grad:
            dp = np.zeros_like(X)
            sq = np.sqrt(x1 / g)
            dp[:, 0] = -0.5 * np.sqrt(g / x1) * (1 - 0) + g * (-0.5 / np.sqrt(x1 * g))
            # Simplify: dp/dx1 = -0.5 * sqrt(g/x1)
            dp[:, 0] = -0.5 * np.sqrt(g / x1)
            dg = 9 / (n - 1)
            for j in range(1, n):
                dp[:, j] = dg * (1 - sq) + g * (0.5 * x1 / g**2) / np.sqrt(x1 / g) * dg
            return p, dp
        return p


funObjZDT12 = FunObjZDT12()
