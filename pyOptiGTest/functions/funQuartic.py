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


class FunQuartic(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Quartic function
        global minimum : f(x)~0 for xi=(0,...,0) (stochastic)
        Design space: -1.28<xi<1.28
        """
        dim = X.shape[1]
        idx = np.arange(1, dim + 1)
        pa = X**4 * idx[np.newaxis, :]
        p = np.sum(pa, axis=1) + np.random.randn(X.shape[0])

        if grad:
            dp = 4 * X**3 * idx[np.newaxis, :]

        return (p, dp) if grad else p



funQuartic = FunQuartic()
