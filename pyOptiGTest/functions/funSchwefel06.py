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


class FunSchwefel06(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schwefel 6 function
        1 global minimum: f(1,3)=0
        Design space: -100<xi<100
        """
        a = 2
        b = 5
        c = 7

        x = X[:, 0]
        y = X[:, 1]
        pa = x + a * y - c
        pb_val = a * x + y - b

        pTmp = np.column_stack([pa, pb_val])

        Ip = np.argmax(np.abs(pTmp), axis=1)
        p = np.max(np.abs(pTmp), axis=1)

        if grad:
            dp = np.zeros_like(X)
            spa = np.sign(pa)
            spb = np.sign(pb_val)

            mask0 = Ip == 0
            mask1 = Ip == 1
            dp[mask0, 0] = spa[mask0]
            dp[mask0, 1] = a * spa[mask0]
            dp[mask1, 0] = a * spb[mask1]
            dp[mask1, 1] = spb[mask1]

        return (p, dp) if grad else p


funSchwefel06 = FunSchwefel06()
