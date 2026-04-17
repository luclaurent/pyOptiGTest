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


class FunShubert1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Shubert 1 function
        1 global minimum: f(7.0835,4.8580)=-186.7309
        Design space: -10<xi<10
        """
        a = 5

        x = X[:, 0]
        y = X[:, 1]
        lI = np.arange(1, a + 1, dtype=float)  # 1..5

        # h[i,k] = lI[k] * cos((lI[k]+1)*x[i] + lI[k])
        hx = lI[np.newaxis, :] * np.cos((lI[np.newaxis, :] + 1) * x[:, np.newaxis] + lI[np.newaxis, :])
        gy = lI[np.newaxis, :] * np.cos((lI[np.newaxis, :] + 1) * y[:, np.newaxis] + lI[np.newaxis, :])

        p = np.sum(hx, axis=1) * np.sum(gy, axis=1)

        if grad:
            dhx = -lI[np.newaxis, :] * (lI[np.newaxis, :] + 1) * np.sin((lI[np.newaxis, :] + 1) * x[:, np.newaxis] + lI[np.newaxis, :])
            dgy = -lI[np.newaxis, :] * (lI[np.newaxis, :] + 1) * np.sin((lI[np.newaxis, :] + 1) * y[:, np.newaxis] + lI[np.newaxis, :])

            dp = np.zeros_like(X)
            dp[:, 0] = np.sum(dhx, axis=1) * np.sum(gy, axis=1)
            dp[:, 1] = np.sum(hx, axis=1) * np.sum(dgy, axis=1)

        return (p, dp) if grad else p


funShubert1 = FunShubert1()
