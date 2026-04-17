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


class FunObjZDT62(TestFunction):
    def evaluate(self, X, grad=False):
        """
        ZDT6 2nd objective function.
        f2(x) = g(x) * [1 - (f1(x)/g(x))^2]
        where g(x) = 1 + 9*(sum(x2..xn)/(n-1))^0.25
        Design space: 0 <= xi <= 1, dim=10
        """

        n = X.shape[1]
        x1 = X[:, 0]
        f1 = 1 - np.exp(-4 * x1) * np.sin(6 * np.pi * x1)**6
        s = np.sum(X[:, 1:], axis=1)
        g = 1 + 9 * (s / (n - 1))**0.25
        p = g * (1 - (f1 / g)**2)

        if grad:
            dp = np.zeros_like(X)
            # Gradient is complex; skip for now
            return p, dp
        return p


funObjZDT62 = FunObjZDT62()
