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


class FunConsConstrEx1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        ConstrEx constraint 1.
        g1(x) = x2 + 9*x1 - 6 >= 0
        """

        p = X[:, 1] + 9 * X[:, 0] - 6

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 9 * np.ones(X.shape[0])
            dp[:, 1] = np.ones(X.shape[0])
            return p, dp
        return p


funConsConstrEx1 = FunConsConstrEx1()
