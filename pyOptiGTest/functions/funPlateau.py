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


class FunPlateau(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Plateau function
        global minimum: f(x)=30 for xi=0
        Design space: -5.12<xi<5.12
        """
        a = 30.0

        pa = np.floor(np.abs(X))
        p = a + np.sum(pa, axis=1)

        if grad:
            dp = np.zeros_like(X)
            is_int = (np.round(X) == X)
            siX = np.sign(X)
            dp[is_int] = 1.0 / np.finfo(float).eps * siX[is_int]

        return (p, dp) if grad else p


funPlateau = FunPlateau()
