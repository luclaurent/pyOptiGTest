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


class FunObjZDT11(TestFunction):
    def evaluate(self, X, grad=False):
        """
        ZDT1 1st objective function.
        f1(x) = x1
        Design space: 0 <= xi <= 1, dim=30
        """

        p = X[:, 0].copy()

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = np.ones(X.shape[0])
            return p, dp
        return p


funObjZDT11 = FunObjZDT11()
