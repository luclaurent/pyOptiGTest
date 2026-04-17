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


class FunCustom06(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Custom function 6
        global minimum : f(x)=-0.4366 for x= 2.9084
        Design space: -1<xi<15
        """
        a = 10
        b = 0
        c = 1
        pv = np.exp(-X / a) * np.cos(c * X) + 1 / a * X + b
        p = np.sum(pv, axis=1)

        if grad:
            dp = -np.exp(-X / a) * (c * np.sin(c * X) + 1 / a * np.cos(c * X)) + 1 / a

        return (p, dp) if grad else p



funCustom06 = FunCustom06()
