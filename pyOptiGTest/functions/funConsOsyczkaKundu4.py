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


class FunConsOsyczkaKundu4(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Osyczka and Kundu constraint 4.
        g4(x) = 2 - x1 + 3*x2 >= 0
        """

        p = 2 - X[:, 0] + 3 * X[:, 1]

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = -np.ones(X.shape[0])
            dp[:, 1] = 3 * np.ones(X.shape[0])
            return p, dp
        return p


funConsOsyczkaKundu4 = FunConsOsyczkaKundu4()
