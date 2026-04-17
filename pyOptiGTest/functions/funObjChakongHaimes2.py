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


class FunObjChakongHaimes2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Chakong and Haimes 2nd objective function.
        f2(x) = 9*x1 - (x2-1)^2
        Design space: -20 <= xi <= 20
        """

        a = 9
        b = 1
        p = a * X[:, 0] - (X[:, 1] - b)**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = a * np.ones(X.shape[0])
            dp[:, 1] = -2 * (X[:, 1] - b)
            return p, dp
        return p


funObjChakongHaimes2 = FunObjChakongHaimes2()
