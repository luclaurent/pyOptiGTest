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


class FunPowell(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Powell function
        global minimum : f(x)=0 for xi=0
        Design space: -4<xi<5
        """
        a = 10
        b = 5
        c = 2
        d = 10

        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        v = X[:, 3]

        pa = z + a * x
        pb = y - v
        pc = x - c * y
        pd = z - v

        p = pa**2 + b * pb**2 + pc**4 + d * pd**4

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * a * pa + 4 * pc**3
            dp[:, 1] = 2 * b * pb - 4 * c * pc**3
            dp[:, 2] = 2 * pa + 4 * d * pd**3
            dp[:, 3] = -2 * b * pb - 4 * d * pd**3

        return (p, dp) if grad else p



funPowell = FunPowell()
