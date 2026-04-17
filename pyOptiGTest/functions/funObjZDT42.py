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


class FunObjZDT42(TestFunction):
    def evaluate(self, X, grad=False):
        """
        ZDT4 2nd objective function.
        f2(x) = g(x) * [1 - sqrt(x1/g(x))]
        where g(x) = 1 + 10*(n-1) + sum(xi^2 - 10*cos(4*pi*xi)) for i=2..n
        Design space: 0 <= x1 <= 1, -5 <= xi <= 5 (i>=2), dim=10
        """

        n = X.shape[1]
        x1 = X[:, 0]
        g = 1 + 10 * (n - 1) + np.sum(X[:, 1:]**2 - 10 * np.cos(4 * np.pi * X[:, 1:]), axis=1)
        p = g * (1 - np.sqrt(x1 / g))

        if grad:
            dp = np.zeros_like(X)
            sq = np.sqrt(x1 / g)
            dp[:, 0] = -0.5 * np.sqrt(g / x1)
            for j in range(1, n):
                dgj = 2 * X[:, j] + 40 * np.pi * np.sin(4 * np.pi * X[:, j])
                dp[:, j] = dgj * (1 - 0.5 * sq - 0.5 * x1 / (g * sq))
            return p, dp
        return p


funObjZDT42 = FunObjZDT42()
