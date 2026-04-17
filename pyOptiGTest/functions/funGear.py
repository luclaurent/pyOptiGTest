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


class FunGear(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Gear function
        Design space: 12<xi<60
        """
        a = 1.0
        b = 6.931

        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        w = X[:, 3]

        fx = np.floor(x)
        fy = np.floor(y)
        fz = np.floor(z)
        fw = np.floor(w)

        p = (a / b - (fx * fy) / (fw * fz)) ** 2

        if grad:
            dp = np.zeros_like(X)
            is_int = (np.imag(X) == 0) & (np.round(X) == X)
            dp[is_int] = 1.0 / np.finfo(float).eps

        return (p, dp) if grad else p


funGear = FunGear()
