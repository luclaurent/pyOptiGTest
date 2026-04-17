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


class FunObjViennet2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Viennet 2nd objective function.
        f2(x) = (3*x1-2*x2+4)^2/8 + (x1-x2+1)^2/27 + 15
        Design space: -3 <= xi <= 3, dim=2
        """

        x1 = X[:, 0]
        x2 = X[:, 1]
        a = 3 * x1 - 2 * x2 + 4
        b = x1 - x2 + 1
        p = a**2 / 8 + b**2 / 27 + 15

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * a * 3 / 8 + 2 * b / 27
            dp[:, 1] = 2 * a * (-2) / 8 + 2 * b * (-1) / 27
            return p, dp
        return p


funObjViennet2 = FunObjViennet2()
