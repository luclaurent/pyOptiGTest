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


class FunCustom01(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Custom function 1
        global minimum : f(x)=5 for x={pi;3*pi}
        Design space: -1<xi<15
        """
        a = 15
        b = 20
        pv = a * np.cos(X) + b
        p = np.sum(pv, axis=1)

        if grad:
            dp = -a * np.sin(X)

        return (p, dp) if grad else p



funCustom01 = FunCustom01()
