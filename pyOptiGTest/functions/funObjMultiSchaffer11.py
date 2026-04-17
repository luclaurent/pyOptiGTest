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


class FunObjMultiSchaffer11(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schaffer N.1 multi-objective 1st objective function.
        f1(x) = x^2
        Design space: -10 <= x <= 10, dim=1
        """

        p = X[:, 0]**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * X[:, 0]
            return p, dp
        return p


funObjMultiSchaffer11 = FunObjMultiSchaffer11()
