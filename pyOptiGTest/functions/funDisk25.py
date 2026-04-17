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


class FunDisk25(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Disk constraint for BirdDisk problem.
        g(x) = (x1+5)^2 + (x2+5)^2 - 25 < 0
        """

        p = (X[:, 0] + 5)**2 + (X[:, 1] + 5)**2 - 25

        if grad:
            dp = 2 * (X + 5)
            return p, dp
        return p


funDisk25 = FunDisk25()
