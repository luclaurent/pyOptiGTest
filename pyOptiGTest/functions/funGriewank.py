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


class FunGriewank(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Griewank function
        global minimum : f(x1,x2)=0 for (0,...,0)
        Design space: -100<xi<100
        """
        a = 1 / 4000
        b = 1

        dim = X.shape[1]
        div = np.sqrt(np.arange(1, dim + 1))
        pa = X / div[np.newaxis, :]
        p = a * np.sum(X**2, axis=1) - np.prod(np.cos(pa), axis=1) + b

        if grad:
            # Gradient: prod of cos except i-th, times sin(pa_i)/div_i + 2a*x_i
            cos_pa = np.cos(pa)
            full_prod = np.prod(cos_pa, axis=1)  # shape (n,)
            # For each variable, prod_except_i = full_prod / cos(pa_i)
            dp = 2 * a * X + np.sin(pa) / div[np.newaxis, :] * (full_prod[:, np.newaxis] / (cos_pa + 1e-300))

        return (p, dp) if grad else p



funGriewank = FunGriewank()
