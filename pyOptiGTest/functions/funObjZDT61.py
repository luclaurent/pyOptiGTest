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


class FunObjZDT61(TestFunction):
    def evaluate(self, X, grad=False):
        """
        ZDT6 1st objective function.
        f1(x) = 1 - exp(-4*x1) * sin(6*pi*x1)^6
        Design space: 0 <= xi <= 1, dim=10
        """

        x1 = X[:, 0]
        p = 1 - np.exp(-4 * x1) * np.sin(6 * np.pi * x1)**6

        if grad:
            dp = np.zeros_like(X)
            s = np.sin(6 * np.pi * x1)
            c = np.cos(6 * np.pi * x1)
            dp[:, 0] = 4 * np.exp(-4 * x1) * s**6 - np.exp(-4 * x1) * 6 * s**5 * c * 6 * np.pi
            return p, dp
        return p


funObjZDT61 = FunObjZDT61()
