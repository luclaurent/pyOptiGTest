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


class FunShubert4(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Shubert 4 function
        1 global minimum: f(-0.80,7.08)=-29.016
        Design space: -10<xi<10
        """
        a = 5
        nbvar = X.shape[1]
        lI = np.arange(1, a + 1, dtype=float)

        p = np.zeros(X.shape[0])
        for itX in range(nbvar):
            h = lI[np.newaxis, :] * np.cos((lI[np.newaxis, :] + 1) * X[:, itX:itX+1] + lI[np.newaxis, :])
            p = p - np.sum(h, axis=1)

        if grad:
            dp = np.zeros_like(X)
            for itX in range(nbvar):
                dh = -lI[np.newaxis, :] * (lI[np.newaxis, :] + 1) * np.sin((lI[np.newaxis, :] + 1) * X[:, itX:itX+1] + lI[np.newaxis, :])
                dp[:, itX] = -np.sum(dh, axis=1)

        return (p, dp) if grad else p


funShubert4 = FunShubert4()
