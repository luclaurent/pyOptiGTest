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


class FunObjViennet3(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Viennet 3rd objective function.
        f3(x) = 1/(x1^2+x2^2+1) - 1.1*exp(-(x1^2+x2^2))
        Design space: -3 <= xi <= 3, dim=2
        """

        x1 = X[:, 0]
        x2 = X[:, 1]
        s = x1**2 + x2**2
        p = 1 / (s + 1) - 1.1 * np.exp(-s)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = -2 * x1 / (s + 1)**2 + 2.2 * x1 * np.exp(-s)
            dp[:, 1] = -2 * x2 / (s + 1)**2 + 2.2 * x2 * np.exp(-s)
            return p, dp
        return p


funObjViennet3 = FunObjViennet3()
