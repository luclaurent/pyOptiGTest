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


class FunStep(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Step function
        global minimum: f(xi)=dim*0.25 for xi=0
        Design space: -5<xi<5
        """
        a = 0.5

        pa = np.floor(X) + a
        pb = pa ** 2
        p = np.sum(pb, axis=1)

        if grad:
            dp = np.zeros_like(X)
            is_int = (np.round(X) == X)
            dp[is_int] = 1.0 / np.finfo(float).eps

        return (p, dp) if grad else p


funStep = FunStep()
