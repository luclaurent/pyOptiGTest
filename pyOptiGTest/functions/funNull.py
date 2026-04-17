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


class FunNull(TestFunction):
    def evaluate(self, X, grad=False):
        """Null function: f(x) = 0"""
        p = np.zeros(X.shape[0])

        if grad:
            dp = np.zeros_like(X)

        return (p, dp) if grad else p



funNull = FunNull()
