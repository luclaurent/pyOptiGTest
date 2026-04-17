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


class FunObjKornBinh1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Binh and Korn 1st objective function.
        f1(x) = 4*(x1^2 + x2^2)
        Design space: 0 <= x1 <= 5, 0 <= x2 <= 3
        """

        a = 4
        p = a * (X[:, 0]**2 + X[:, 1]**2)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * a * X[:, 0]
            dp[:, 1] = 2 * a * X[:, 1]
            return p, dp
        return p


funObjKornBinh1 = FunObjKornBinh1()
