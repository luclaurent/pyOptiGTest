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


class FunConsSimionescu(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Simionescu constraint function.
        g(x) = x1^2 + x2^2 - (rt + rs*cos(n*atan2(x1,x2)))^2 <= 0
        """

        xxx = X[:, 0]
        yyy = X[:, 1]

        rt = 1.0
        rs = 0.2
        n = 8

        t = np.arctan2(xxx, yyy)
        td = rt + rs * np.cos(n * t)
        p = xxx**2 + yyy**2 - td**2

        if grad:
            dp = np.zeros_like(X)
            dtx = yyy / (xxx**2 + yyy**2)
            dty = -xxx / (xxx**2 + yyy**2)
            dp[:, 0] = 2 * xxx + 2 * n * rs * dtx * np.sin(n * t) * td
            dp[:, 1] = 2 * yyy + 2 * n * rs * dty * np.sin(n * t) * td
            return p, dp
        return p


funConsSimionescu = FunConsSimionescu()
