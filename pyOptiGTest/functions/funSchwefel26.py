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


class FunSchwefel26(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schwefel 26 function
        1 global minimum: x=(420.9687,...) >> f(x)=0
        Design space: -500<xi<500
        """
        a = 418.9829
        nbvar = X.shape[1]

        pa = np.abs(X)
        sqa = np.sqrt(pa)
        ssqa = np.sin(sqa)
        pb_val = X * ssqa

        p = a * nbvar - np.sum(pb_val, axis=1)

        if grad:
            csqa = np.cos(sqa)
            dp = -ssqa - X * np.sign(X) * csqa / (2 * sqa)
            dp[np.abs(X) < np.finfo(float).eps] = 0.0

        return (p, dp) if grad else p


funSchwefel26 = FunSchwefel26()
