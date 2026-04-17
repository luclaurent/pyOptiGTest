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


class FunStepInt(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Step Int function
        global minimum: f(xi)=25 for xi=0
        Design space: -5<xi<5
        """
        a = 25.0

        pa = np.floor(X)
        p = a + np.sum(pa, axis=1)

        if grad:
            dp = np.zeros_like(X)
            is_int = (np.round(X) == X)
            dp[is_int] = 1.0 / np.finfo(float).eps

        return (p, dp) if grad else p


funStepInt = FunStepInt()
