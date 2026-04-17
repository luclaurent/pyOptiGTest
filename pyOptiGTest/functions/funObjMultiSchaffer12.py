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


class FunObjMultiSchaffer12(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schaffer N.1 multi-objective 2nd objective function.
        f2(x) = (x-2)^2
        Design space: -10 <= x <= 10, dim=1
        """

        a = 2
        p = (X[:, 0] - a)**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * (X[:, 0] - a)
            return p, dp
        return p


funObjMultiSchaffer12 = FunObjMultiSchaffer12()
