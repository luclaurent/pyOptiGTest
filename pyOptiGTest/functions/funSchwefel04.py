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


class FunSchwefel04(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schwefel 4 function (Rosenbrock-like)
        1 minimum global: f(1,...,1)=0
        Design space 0<xi<10
        """
        a = 1

        x1 = X[:, 0]
        pa = X - a
        pb = x1[:, np.newaxis] - X**2

        p = np.sum(pa**2 + pb**2, axis=1)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * (x1 - a) + 2 * (1 - 2 * x1) * (x1 - x1**2) + 2 * np.sum(pb[:, 1:], axis=1)
            dp[:, 1:] = 2 * pa[:, 1:] - 4 * X[:, 1:] * pb[:, 1:]

        return (p, dp) if grad else p



funSchwefel04 = FunSchwefel04()
