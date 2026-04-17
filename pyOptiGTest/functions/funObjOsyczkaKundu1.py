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


class FunObjOsyczkaKundu1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Osyczka and Kundu 1st objective function.
        f1(x) = -[25*(x1-2)^2 + (x2-2)^2 + (x3-1)^2 * (x4-4)^2 + (x5-1)^2]
        Design space: 0<=x1<=10, 0<=x2<=10, 1<=x3<=5, 0<=x4<=6, 1<=x5<=5, 0<=x6<=10
        """

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]

        p = -(25 * (x1 - 2)**2 + (x2 - 2)**2 + (x3 - 1)**2 * (x4 - 4)**2 + (x5 - 1)**2)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = -50 * (x1 - 2)
            dp[:, 1] = -2 * (x2 - 2)
            dp[:, 2] = -2 * (x3 - 1) * (x4 - 4)**2
            dp[:, 3] = -2 * (x3 - 1)**2 * (x4 - 4)
            dp[:, 4] = -2 * (x5 - 1)
            return p, dp
        return p


funObjOsyczkaKundu1 = FunObjOsyczkaKundu1()
