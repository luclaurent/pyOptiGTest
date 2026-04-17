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


class FunObjCTP11(TestFunction):
    def evaluate(self, X, grad=False):
        """
        CTP1 1st objective function.
        f1(x) = x1
        Design space: 0 <= x1 <= 1, 0 <= x2 <= 1
        """

        p = X[:, 0].copy()

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = np.ones(X.shape[0])
            return p, dp
        return p


funObjCTP11 = FunObjCTP11()
