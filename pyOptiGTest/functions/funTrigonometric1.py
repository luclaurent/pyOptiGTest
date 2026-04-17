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


class FunTrigonometric1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Trigonometric 1 function
        1 global minimum: f(0,...,0)=0
        Design space: 0<xi<pi
        """
        nbvar = X.shape[1]

        cx = np.cos(X)
        sx = np.sin(X)

        lI = np.arange(1, nbvar + 1, dtype=float)[np.newaxis, :]

        sumcx = np.sum(cx + lI * (1 - cx - sx), axis=1)

        pa = nbvar - sumcx[:, np.newaxis]

        p = np.sum(pa ** 2, axis=1)

        if grad:
            dp = 2 * pa * (sx + lI * (-sx + cx)) * nbvar

        return (p, dp) if grad else p


funTrigonometric1 = FunTrigonometric1()
