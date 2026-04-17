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


class FunHartmann3(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Hartmann 3 function
        global minimum: f(x)=-3.86278 for x=[0.114614,0.555649,0.852547]
        Design space: 0<xi<1
        """
        ma = np.array([[3.0, 0.1, 3.0, 0.1],
                       [10.0, 10.0, 10.0, 10.0],
                       [30.0, 35.0, 30.0, 35.0]])
        mp = np.array([[0.36890, 0.46990, 0.10910, 0.03815],
                       [0.11700, 0.43870, 0.87320, 0.57430],
                       [0.26730, 0.74700, 0.55470, 0.88280]])
        c = np.array([1.0, 1.2, 3.0, 3.2])
        b = 4
        dim = 3

        XX = X[:, :dim]
        n_samples = XX.shape[0]
        p = np.zeros(n_samples)
        for it in range(b):
            pa = XX - mp[:dim, it][np.newaxis, :]
            pb_val = ma[:dim, it][np.newaxis, :] * pa**2
            p = p - c[it] * np.exp(-np.sum(pb_val, axis=1))

        if grad:
            dp = np.zeros_like(X)
            for it in range(b):
                pa = XX - mp[:dim, it][np.newaxis, :]
                pb_val = ma[:dim, it][np.newaxis, :] * pa**2
                pc = np.exp(-np.sum(pb_val, axis=1))
                pe = ma[:dim, it][np.newaxis, :] * pa
                for itD in range(dim):
                    dp[:, itD] = dp[:, itD] + 2 * c[it] * pe[:, itD] * pc

        return (p, dp) if grad else p



funHartmann3 = FunHartmann3()
