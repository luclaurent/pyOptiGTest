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


class FunTripod(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Tripod function
        1 global minimum: f(0,-50)=0
        Design space: -100<xi<100
        """
        a = 1.0
        b = 50.0
        c = 2.0

        x = X[:, 0]
        y = X[:, 1]

        # ps(x) = 1 if x>=0, else 0
        pX = np.where(x >= 0, 1.0, 0.0)
        pY = np.where(y >= 0, 1.0, 0.0)

        pa = pY * (a + pX)
        pb = x + b * pY * (a - c * pX)
        pc = y + b * (a - c * pY)

        p = pa + np.abs(pb) + np.abs(pc)

        if grad:
            dp = np.zeros_like(X)
            eps_val = np.finfo(float).eps
            dpX = np.where(x == 0, 1.0 / eps_val, 0.0)
            dpY = np.where(y == 0, 1.0 / eps_val, 0.0)

            dp[:, 0] = pY * dpX + np.sign(pb) * (1 - b * c * pY * dpX)
            dp[:, 1] = ((a + pX) * dpY
                         + b * np.sign(pb) * dpY * (a - c * pX)
                         + np.sign(pc) * (1 - b * c * dpY))

        return (p, dp) if grad else p


funTripod = FunTripod()
