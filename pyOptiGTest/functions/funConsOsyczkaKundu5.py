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


class FunConsOsyczkaKundu5(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Osyczka and Kundu constraint 5.
        g5(x) = 4 - (x3-3)^2 - x4 >= 0
        """

        p = 4 - (X[:, 2] - 3)**2 - X[:, 3]

        if grad:
            dp = np.zeros_like(X)
            dp[:, 2] = -2 * (X[:, 2] - 3)
            dp[:, 3] = -np.ones(X.shape[0])
            return p, dp
        return p


funConsOsyczkaKundu5 = FunConsOsyczkaKundu5()
