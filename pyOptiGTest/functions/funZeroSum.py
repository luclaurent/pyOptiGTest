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


class FunZeroSum(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Zero Sum function
        1 global minimum: f(x)=0 such that sum(xi)=0
        Design space: -10<xi<10
        """
        a = 1.0
        b = 1e4
        c = 0.5

        sumX = np.sum(X, axis=1)

        p = a + (b * np.abs(sumX)) ** c
        IX = sumX == 0
        p[IX] = 0.0

        if grad:
            dp = np.full_like(X, c * b ** c * np.sign(sumX)[:, np.newaxis] * np.abs(sumX)[:, np.newaxis] ** (c - 1))

        return (p, dp) if grad else p


funZeroSum = FunZeroSum()
