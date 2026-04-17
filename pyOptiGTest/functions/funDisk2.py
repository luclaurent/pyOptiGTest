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


class FunDisk2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Disk constraint for RosenbrockDisk problem.
        g(x) = x1^2 + x2^2 - 2 <= 0
        """

        xxx = X[:, 0]
        yyy = X[:, 1]

        p = xxx**2 + yyy**2 - 2

        if grad:
            dp = 2 * X
            return p, dp
        return p


funDisk2 = FunDisk2()
