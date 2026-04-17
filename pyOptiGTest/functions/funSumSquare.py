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


class FunSumSquare(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Sum Square function
        1 global minimum: x=(0,...,0) >> f(x)=0
        Design space: -10<xi<10
        """
        nbvar = X.shape[1]
        nu = np.arange(1, nbvar + 1, dtype=float)[np.newaxis, :]

        cal = nu * X ** 2
        p = np.sum(cal, axis=1)

        if grad:
            dp = 2 * nu * X

        return (p, dp) if grad else p


funSumSquare = FunSumSquare()
