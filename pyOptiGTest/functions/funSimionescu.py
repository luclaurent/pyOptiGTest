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


class FunSimionescu(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Simionescu function (objective function for constrained problem).
        L. LAURENT -- 02/05/2018 -- luc.laurent@lecnam.net
        global minimum : f(x1,x2)=-0.072 at (+-0.84852813, -+0.84852813)
        Design space: -1.25 < x1 < 1.25, -1.25 < x2 < 1.25
        """

        xxx = X[:, 0]
        yyy = X[:, 1]

        a = 0.1
        p = a * xxx * yyy

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = a * yyy
            dp[:, 1] = a * xxx
            return p, dp
        return p


funSimionescu = FunSimionescu()
