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


class FunCustom07(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Custom function 7
        global minimum : f(x)=20.6508 for x= 9.6952
        Design space: -1<xi<15
        """
        a10 = 0.002201370
        a9 = -0.1052876
        a8 = 2.151650
        a7 = -24.60697
        a6 = 173.4160
        a5 = -782.1379
        a4 = 2267.874
        a3 = -4114.980
        a2 = 4357.030
        a1 = -2327.900
        b = 550

        pv = (a10*X**10 + a9*X**9 + a8*X**8 + a7*X**7 + a6*X**6
              + a5*X**5 + a4*X**4 + a3*X**3 + a2*X**2 + a1*X + b)
        p = np.sum(pv, axis=1)

        if grad:
            dp = (10*a10*X**9 + 9*a9*X**8 + 8*a8*X**7 + 7*a7*X**6 + 6*a6*X**5
                  + 5*a5*X**4 + 4*a4*X**3 + 3*a3*X**2 + 2*a2*X + a1)

        return (p, dp) if grad else p



funCustom07 = FunCustom07()
