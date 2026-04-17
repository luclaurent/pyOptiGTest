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


class FunCustom04(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Custom function 4
        global minimum : f(x)=0 for x=0
        Design space: -1<xi<15
        """
        a = 2
        pv = X**a
        p = np.sum(pv, axis=1)

        if grad:
            dp = a * X**(a - 1)

        return (p, dp) if grad else p



funCustom04 = FunCustom04()
