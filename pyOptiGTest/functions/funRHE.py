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


class FunRHE(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Rotated Hyper-Ellipsoid function
        1 global minimum: x=(0,...,0) >> f(x)=0
        Design space: -65.536<xi<65.536
        """
        nbvar = X.shape[1]
        coef = np.arange(nbvar, 0, -1, dtype=float)[np.newaxis, :]

        cal = coef * X ** 2
        p = np.sum(cal, axis=1)

        if grad:
            dp = 2 * coef * X

        return (p, dp) if grad else p


funRHE = FunRHE()
