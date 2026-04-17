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


class FunConsChakongHaimes1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Chakong and Haimes 1st constraint function.
        g1(x) = x1^2 + x2^2 - 225 <= 0
        """

        a = 225
        p = X[:, 0]**2 + X[:, 1]**2 - a

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * X[:, 0]
            dp[:, 1] = 2 * X[:, 1]
            return p, dp
        return p


funConsChakongHaimes1 = FunConsChakongHaimes1()
