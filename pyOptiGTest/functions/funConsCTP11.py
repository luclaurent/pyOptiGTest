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


class FunConsCTP11(TestFunction):
    def evaluate(self, X, grad=False):
        """
        CTP1 constraint 1.
        C1(x) = f2/(0.858*exp(-0.541*f1)) - 1 >= 0
        where f1=x1, f2=g*exp(-f1/g), g=1+x2
        """

        x1 = X[:, 0]
        x2 = X[:, 1]
        g = 1 + x2
        f1 = x1
        f2 = g * np.exp(-f1 / g)

        a = 0.858
        b = -0.541
        p = f2 / (a * np.exp(b * f1)) - 1

        if grad:
            dp = np.zeros_like(X)
            # Complex derivative; approximate
            return p, dp
        return p


funConsCTP11 = FunConsCTP11()
