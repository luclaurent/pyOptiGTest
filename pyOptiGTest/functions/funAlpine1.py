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


class FunAlpine1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Alpine's function 1
        1 global minimum : x=(0,0,...,0) >> f(x)=0
        design space -10<xi<10
        """
        a = 0.1
        fx = X * np.sin(X) + a * X
        p = np.sum(np.abs(fx), axis=1)

        if grad:
            dp = np.sign(fx) * (np.sin(X) + 0.1 + X * np.cos(X))

        return (p, dp) if grad else p



funAlpine1 = FunAlpine1()
