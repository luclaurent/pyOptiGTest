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


class FunNeedleEye(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Needle Eye function
        global minimum: f(x)=1 for xi=0
        Design space: -10<xi<10
        """
        a = 100.0
        b = 1e-4

        aX = np.abs(X)
        pa = a + aX
        IXa = aX < b
        IXb = aX > b
        IXAll = np.all(IXa, axis=1)

        p = np.sum(pa * IXb, axis=1)
        p[IXAll] = 1.0

        if grad:
            dp = np.sign(X) * IXb
            dp[IXAll, :] = 0.0

        return (p, dp) if grad else p


funNeedleEye = FunNeedleEye()
