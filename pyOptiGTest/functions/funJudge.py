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


class FunJudge(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Judge function
        global minimum : f(x)=16.0817307 for x=[0.86479 1.2357]
        Design space: -10<xi<10
        """
        m = 20
        C = np.array([4.284, 4.149, 3.877, 0.533, 2.211, 2.389, 2.145, 3.231, 1.998, 1.379,
                      2.106, 1.428, 1.011, 2.179, 2.858, 1.388, 1.651, 1.593, 1.046, 2.152])
        B = np.array([0.286, 0.973, 0.384, 0.276, 0.973, 0.543, 0.957, 0.948, 0.543, 0.797,
                      0.936, 0.889, 0.006, 0.828, 0.399, 0.617, 0.939, 0.784, 0.072, 0.889])
        A = np.array([0.645, 0.585, 0.310, 0.058, 0.455, 0.779, 0.259, 0.202, 0.028, 0.099,
                      0.142, 0.296, 0.175, 0.180, 0.842, 0.039, 0.103, 0.620, 0.158, 0.704])

        # Broadcasting: C,B,A are (m,), x,y are (n,)
        x = X[:, 0]
        y = X[:, 1]

        # pa shape: (n, m)
        pa = x[:, np.newaxis] + B[np.newaxis, :] * y[:, np.newaxis] + A[np.newaxis, :] * y[:, np.newaxis]**2 - C[np.newaxis, :]
        p = np.sum(pa**2, axis=1)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = np.sum(2 * pa, axis=1)
            dp[:, 1] = np.sum(2 * (B[np.newaxis, :] + 2 * A[np.newaxis, :] * y[:, np.newaxis]) * pa, axis=1)

        return (p, dp) if grad else p



funJudge = FunJudge()
