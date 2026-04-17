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


class FunObjPoloni2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Poloni two-objective 2nd objective function.
        f2(x) = (x1+3)^2 + (x2+1)^2
        Design space: -pi <= xi <= pi
        """

        a = 3
        b = 1
        p = (X[:, 0] + a)**2 + (X[:, 1] + b)**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * (X[:, 0] + a)
            dp[:, 1] = 2 * (X[:, 1] + b)
            return p, dp
        return p


funObjPoloni2 = FunObjPoloni2()
