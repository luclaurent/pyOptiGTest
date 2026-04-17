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


class FunSchwefel21(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schwefel 21 function
        1 global minimum: f(0,...,0)=0
        Design space: -100<xi<100
        """
        pa = np.abs(X)
        Ip = np.argmax(pa, axis=1)
        p = np.max(pa, axis=1)

        if grad:
            dpTmp = np.sign(X)
            dp = np.zeros_like(X)
            for i in range(X.shape[0]):
                dp[i, Ip[i]] = dpTmp[i, Ip[i]]

        return (p, dp) if grad else p


funSchwefel21 = FunSchwefel21()
