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


class FunStochastic(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Stochastic function
        1 global minimum: f(1,1/2,...,1/n)=0
        Design space: -5<xi<5
        """
        nbvar = X.shape[1]
        lI = np.arange(1, nbvar + 1, dtype=float)[np.newaxis, :]

        pa = X - 1.0 / lI
        rEps = np.random.rand(*X.shape)
        pb_val = rEps * np.abs(pa)

        p = np.sum(pb_val, axis=1)

        if grad:
            dp = rEps * np.sign(pa)

        return (p, dp) if grad else p


funStochastic = FunStochastic()
