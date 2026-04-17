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


class FunObjKursawe1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Kursawe 1st objective function.
        f1(x) = sum_{i=1}^{n-1} [-10 * exp(-0.2 * sqrt(x_i^2 + x_{i+1}^2))]
        Design space: -5 <= xi <= 5, dim=3
        """

        a = -10
        b = -0.2
        n = X.shape[1]
        p = np.zeros(X.shape[0])
        for i in range(n - 1):
            p += a * np.exp(b * np.sqrt(X[:, i]**2 + X[:, i + 1]**2))

        if grad:
            dp = np.zeros_like(X)
            for i in range(n - 1):
                r = np.sqrt(X[:, i]**2 + X[:, i + 1]**2)
                e = a * b * np.exp(b * r)
                mask = r > 0
                dp[mask, i] += e[mask] * X[mask, i] / r[mask]
                dp[mask, i + 1] += e[mask] * X[mask, i + 1] / r[mask]
            return p, dp
        return p


funObjKursawe1 = FunObjKursawe1()
