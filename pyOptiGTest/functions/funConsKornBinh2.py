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


class FunConsKornBinh2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Binh and Korn 2nd constraint function.
        g2(x) = (x1-8)^2 + (x2+3)^2 - 7.7 >= 0
        """

        a = 8
        b = 3
        c = 7.7
        p = (X[:, 0] - a)**2 + (X[:, 1] + b)**2 - c

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * (X[:, 0] - a)
            dp[:, 1] = 2 * (X[:, 1] + b)
            return p, dp
        return p


funConsKornBinh2 = FunConsKornBinh2()
