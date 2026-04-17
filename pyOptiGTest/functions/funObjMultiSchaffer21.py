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


class FunObjMultiSchaffer21(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schaffer N.2 multi-objective 1st objective function.
        f1(x) = x <= 1: -x
                1 < x <= 3: x - 2
                3 < x <= 4: 4 - x
                x > 4: x - 4
        Design space: -5 <= x <= 10, dim=1
        """

        x = X[:, 0]
        p = np.zeros_like(x)

        mask1 = x <= 1
        mask2 = (x > 1) & (x <= 3)
        mask3 = (x > 3) & (x <= 4)
        mask4 = x > 4

        p[mask1] = -x[mask1]
        p[mask2] = x[mask2] - 2
        p[mask3] = 4 - x[mask3]
        p[mask4] = x[mask4] - 4

        if grad:
            dp = np.zeros_like(X)
            dp[mask1, 0] = -1
            dp[mask2, 0] = 1
            dp[mask3, 0] = -1
            dp[mask4, 0] = 1
            return p, dp
        return p


funObjMultiSchaffer21 = FunObjMultiSchaffer21()
