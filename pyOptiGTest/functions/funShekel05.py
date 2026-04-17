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


class FunShekel05(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Shekel 5 function
        1 global minimum: f(4,4,4,4)=-10.1532
        Design space: 0<xi<10
        """
        a = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7]], dtype=float)
        c = 0.1 * np.array([1, 2, 2, 4, 6], dtype=float)
        m = 5
        dim = 4

        XX = X[:, :dim]
        n_samples = XX.shape[0]
        p = np.zeros(n_samples)
        for itI in range(m):
            pTmp = XX - a[itI, :][np.newaxis, :]
            hI = c[itI] + np.sum(pTmp**2, axis=1)
            p = p - 1.0 / hI

        if grad:
            dp = np.zeros_like(X)
            for itI in range(m):
                pTmp = XX - a[itI, :][np.newaxis, :]
                hI = c[itI] + np.sum(pTmp**2, axis=1)
                dhI = 2 * pTmp
                dp[:, :dim] = dp[:, :dim] + dhI / hI[:, np.newaxis]**2

        return (p, dp) if grad else p



funShekel05 = FunShekel05()
