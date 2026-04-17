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


class FunConsOsyczkaKundu3(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Osyczka and Kundu constraint 3.
        g3(x) = 2 - x2 + x1 >= 0
        """

        p = 2 - X[:, 1] + X[:, 0]

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = np.ones(X.shape[0])
            dp[:, 1] = -np.ones(X.shape[0])
            return p, dp
        return p


funConsOsyczkaKundu3 = FunConsOsyczkaKundu3()
