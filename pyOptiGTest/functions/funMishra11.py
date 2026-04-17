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


class FunMishra11(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Mishra 11 function
        global minimum : f(x)=0 for xi=0
        Design space: -10<xi<10
        """
        dim = X.shape[1]
        pa = np.sum(np.abs(X), axis=1)
        pb = np.prod(np.abs(X), axis=1)

        pc = 1 / dim * pa - pb**(1 / dim)
        p = pc**2

        if grad:
            sx = np.sign(X)
            dp = np.zeros_like(X)
            for i in range(dim):
                cols = list(range(dim))
                cols.remove(i)
                pd_i = np.prod(np.abs(X[:, cols]), axis=1)
                dp[:, i] = 2 / dim * sx[:, i] * pc * (1 - pd_i * pb**(1 / dim - 1))

        return (p, dp) if grad else p



funMishra11 = FunMishra11()
