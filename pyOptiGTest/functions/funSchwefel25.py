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


class FunSchwefel25(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schwefel 25 function
        1 global minimum: x=(1,...,1) >> f(x)=0
        Design space: 0<xi<10
        """
        a = 1.0

        pa = X - a
        pb_val = X[:, 0:1] - X ** 2

        p = np.sum(pa ** 2 + pb_val ** 2, axis=1)

        if grad:
            dp = 2 * pa - 4 * X * pb_val
            dp[:, 0] = dp[:, 0] + np.sum(2 * pb_val, axis=1)

        return (p, dp) if grad else p


funSchwefel25 = FunSchwefel25()
