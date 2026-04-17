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
from math import factorial as _factorial


class FunMishra07(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Mishra 7 function
        global minimum : f(x)=0 for xi=sqrt(n)
        Design space: -10<xi<10
        """
        dim = X.shape[1]
        pa = X - _factorial(dim)
        pb = np.prod(pa, axis=1)
        p = pb**2

        if grad:
            dp = np.zeros_like(X)
            for i in range(dim):
                cols = list(range(dim))
                cols.remove(i)
                pc = np.prod(pa[:, cols], axis=1)
                dp[:, i] = 2 * pc * pb

        return (p, dp) if grad else p



funMishra07 = FunMishra07()
