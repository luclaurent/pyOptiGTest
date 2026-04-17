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


class FunSodp(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Sum of Different Powers function
        1 global minimum: f(0,...,0)=0
        Design space: -1<xi<1
        """
        nbvar = X.shape[1]
        lI = np.arange(1, nbvar + 1, dtype=float)[np.newaxis, :]

        pX = np.abs(X) ** (lI + 1)
        p = np.sum(pX, axis=1)

        if grad:
            dp = (lI + 1) * np.sign(X) * np.abs(X) ** lI

        return (p, dp) if grad else p


funSodp = FunSodp()
