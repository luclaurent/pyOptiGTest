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


class FunConsOsyczkaKundu2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Osyczka and Kundu constraint 2.
        g2(x) = 6 - x1 - x2 >= 0
        """

        p = 6 - X[:, 0] - X[:, 1]

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = -np.ones(X.shape[0])
            dp[:, 1] = -np.ones(X.shape[0])
            return p, dp
        return p


funConsOsyczkaKundu2 = FunConsOsyczkaKundu2()
