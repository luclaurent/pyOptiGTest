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


class FunPowerSum(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Power Sum function
        global minimum: f(x)=0 for xi=[1,2,2,3]
        Design space: 0<xi<4
        """
        b = np.array([8.0, 18.0, 44.0, 114.0])

        # X shape: (n, 4), liK: powers 1..4
        liK = np.arange(1, 5, dtype=float)  # [1,2,3,4]

        # pa[i,j,k] = X[i,j]^liK[k] -> shape (n, 4, 4)
        pa = X[:, :, np.newaxis] ** liK[np.newaxis, np.newaxis, :]

        # sum over variables (axis=1) for each power k
        sum_pa = np.sum(pa, axis=1)  # (n, 4)

        p = np.sum((sum_pa - b[np.newaxis, :]) ** 2, axis=1)

        if grad:
            dp = np.zeros_like(X)
            for k in range(4):
                dp = dp + 2 * liK[k] * X ** (liK[k] - 1) * (sum_pa[:, k] - b[k])[:, np.newaxis]

        return (p, dp) if grad else p


funPowerSum = FunPowerSum()
