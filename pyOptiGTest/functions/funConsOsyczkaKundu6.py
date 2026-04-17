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


class FunConsOsyczkaKundu6(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Osyczka and Kundu constraint 6.
        g6(x) = (x5-3)^2 + x6 - 4 >= 0
        """

        p = (X[:, 4] - 3)**2 + X[:, 5] - 4

        if grad:
            dp = np.zeros_like(X)
            dp[:, 4] = 2 * (X[:, 4] - 3)
            dp[:, 5] = np.ones(X.shape[0])
            return p, dp
        return p


funConsOsyczkaKundu6 = FunConsOsyczkaKundu6()
