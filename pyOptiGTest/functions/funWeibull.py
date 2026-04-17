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


class FunWeibull(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Weibull function
        global minimum : f(x)=0.326474 for x={41.8923,1.32038,23.0365}
        Design space: 0<xi<100
        """
        aa = 99
        b = 25
        c = 50
        d = 2 / 3
        f = 0.01

        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]

        listI = np.arange(1, aa + 1)
        zi = f * listI
        yi = b + (c * np.log(1 / zi))**d

        pa = yi[np.newaxis, :] - z[:, np.newaxis]
        pb = np.abs(pa)**(y[:, np.newaxis]) / x[:, np.newaxis]
        pc = np.exp(-pb)
        pd = pc - zi[np.newaxis, :]

        p = np.sum(pd**2, axis=1)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * np.sum(pb / x[:, np.newaxis] * pc * pd, axis=1)
            dp[:, 1] = 2 * np.sum(-np.log(np.abs(pa) + 1e-300) / x[:, np.newaxis] * pb * pc * pd, axis=1)
            dp[:, 2] = 2 * np.sum(np.abs(pa)**(y[:, np.newaxis] - 1) / x[:, np.newaxis] * y[:, np.newaxis] * pc * pd, axis=1)

        return (p, dp) if grad else p



funWeibull = FunWeibull()
