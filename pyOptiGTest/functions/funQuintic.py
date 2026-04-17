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


class FunQuintic(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Quintic function
        global minimum : f(x)=0 for xi=(1 or 2)
        Design space: -10<xi<10
        """
        a = 3
        b = 4
        c = 2
        d = 10
        e = 4

        pa = X**5 - a * X**4 + b * X**3 + c * X**2 - d * X - e
        p = np.sum(np.abs(pa), axis=1)

        if grad:
            dp = (5 * X**4 - 4 * a * X**3 + 3 * b * X**2 + 2 * c * X - d) * np.sign(pa)

        return (p, dp) if grad else p



funQuintic = FunQuintic()
