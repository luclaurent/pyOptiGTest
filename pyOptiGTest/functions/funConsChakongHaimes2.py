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


class FunConsChakongHaimes2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Chakong and Haimes 2nd constraint function.
        g2(x) = x1 - 3*x2 + 10 <= 0
        """

        a = 3
        b = 10
        p = X[:, 0] - a * X[:, 1] + b

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = np.ones(X.shape[0])
            dp[:, 1] = -a * np.ones(X.shape[0])
            return p, dp
        return p


funConsChakongHaimes2 = FunConsChakongHaimes2()
