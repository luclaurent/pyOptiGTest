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


class FunObjTestFun42(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Test Function 4 (CTP-like) 2nd objective function.
        f2(x) = (1+x2)/x1
        Design space: 0 <= x1 <= 1, -1 <= x2 <= 1
        """

        x1 = X[:, 0]
        x2 = X[:, 1]
        p = (1 + x2) / x1

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = -(1 + x2) / x1**2
            dp[:, 1] = 1 / x1
            return p, dp
        return p


funObjTestFun42 = FunObjTestFun42()
