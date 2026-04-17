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


class FunTrid(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Trid function
        1 minimum global: f(6,10,12,12,10,6)=-50
        Design space -20<xi<20
        """
        pa = np.sum((X - 1)**2, axis=1)
        pb = np.sum(X[:, 1:] * X[:, :-1], axis=1)
        p = pa - pb

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * (X[:, 0] - 1) - X[:, 1]
            dp[:, -1] = 2 * (X[:, -1] - 1) - X[:, -2]
            if X.shape[1] > 2:
                dp[:, 1:-1] = 2 * (X[:, 1:-1] - 1) - X[:, 2:] - X[:, 0:-2]

        return (p, dp) if grad else p



funTrid = FunTrid()
