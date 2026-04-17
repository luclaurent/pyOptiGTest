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


class FunObjKornBinh2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Binh and Korn 2nd objective function.
        f2(x) = (x1-5)^2 + (x2-5)^2
        Design space: 0 <= x1 <= 5, 0 <= x2 <= 3
        """

        a = 5
        p = (X[:, 0] - a)**2 + (X[:, 1] - a)**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * (X[:, 0] - a)
            dp[:, 1] = 2 * (X[:, 1] - a)
            return p, dp
        return p


funObjKornBinh2 = FunObjKornBinh2()
