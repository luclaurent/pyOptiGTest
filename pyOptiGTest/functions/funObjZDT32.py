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


class FunObjZDT32(TestFunction):
    def evaluate(self, X, grad=False):
        """
        ZDT3 2nd objective function.
        f2(x) = g(x) * [1 - sqrt(x1/g(x)) - (x1/g(x))*sin(10*pi*x1)]
        where g(x) = 1 + 9/(n-1) * sum(x2..xn)
        Design space: 0 <= xi <= 1, dim=30
        """

        n = X.shape[1]
        x1 = X[:, 0]
        g = 1 + 9 / (n - 1) * np.sum(X[:, 1:], axis=1)
        r = x1 / g
        p = g * (1 - np.sqrt(r) - r * np.sin(10 * np.pi * x1))

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = -0.5 * np.sqrt(g / x1) - np.sin(10 * np.pi * x1) - 10 * np.pi * x1 * np.cos(10 * np.pi * x1)
            dg = 9 / (n - 1)
            for j in range(1, n):
                dp[:, j] = dg * (1 - 0.5 * np.sqrt(r) - r * np.sin(10 * np.pi * x1) + 0.5 * x1 / (g * np.sqrt(r)))
            return p, dp
        return p


funObjZDT32 = FunObjZDT32()
