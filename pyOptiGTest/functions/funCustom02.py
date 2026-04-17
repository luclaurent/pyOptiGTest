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


class FunCustom02(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Custom function 2
        global minimum : f(x)=-0.436559 for x=2.9844
        Design space: -1<xi<15
        """
        a = 10
        b = 0
        pv = np.exp(-X / a) * np.cos(X) + 1 / a * X + b
        p = np.sum(pv, axis=1)

        if grad:
            dp = -np.exp(-X / a) * (np.sin(X) + 1 / a * np.cos(X)) + 1 / a

        return (p, dp) if grad else p



funCustom02 = FunCustom02()
