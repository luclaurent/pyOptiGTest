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


class FunObjOsyczkaKundu2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Osyczka and Kundu 2nd objective function.
        f2(x) = x1^2 + x2^2 + x3^2 + x4^2 + x5^2 + x6^2
        Design space: 0<=x1<=10, 0<=x2<=10, 1<=x3<=5, 0<=x4<=6, 1<=x5<=5, 0<=x6<=10
        """

        p = np.sum(X**2, axis=1)

        if grad:
            dp = 2 * X
            return p, dp
        return p


funObjOsyczkaKundu2 = FunObjOsyczkaKundu2()
