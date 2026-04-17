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


class FunObjViennet1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Viennet 1st objective function.
        f1(x) = 0.5*(x1^2+x2^2) + sin(x1^2+x2^2)
        Design space: -3 <= xi <= 3, dim=2
        """

        x1 = X[:, 0]
        x2 = X[:, 1]
        s = x1**2 + x2**2
        p = 0.5 * s + np.sin(s)

        if grad:
            dp = np.zeros_like(X)
            cs = np.cos(s)
            dp[:, 0] = x1 + 2 * x1 * cs
            dp[:, 1] = x2 + 2 * x2 * cs
            return p, dp
        return p


funObjViennet1 = FunObjViennet1()
