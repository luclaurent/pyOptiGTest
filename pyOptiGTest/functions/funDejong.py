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


class FunDejong(TestFunction):
    def evaluate(self, X, grad=False):
        """
        De Jong function (Sphere)
        1 global minimum : x=(0,...,0) >> f(x)=0
        Design space: -5.12<xi<5.12
        """
        p = np.sum(X ** 2, axis=1)

        if grad:
            dp = 2 * X

        return (p, dp) if grad else p


funDejong = FunDejong()
