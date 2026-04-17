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


class FunCustom03(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Custom function 3
        global minimum : f(x)=-1 for x={pi/4;3*pi/4;...}
        Design space: -1<xi<15
        """
        a = 4
        pv = np.cos(a * X)
        p = np.sum(pv, axis=1)

        if grad:
            dp = -a * np.sin(a * X)

        return (p, dp) if grad else p



funCustom03 = FunCustom03()
