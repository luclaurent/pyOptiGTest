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


class FunGulfResearch(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Gulf Research function
        global minimum : f(x1,x2)=0 for (50,25,1.5)
        Design space: 0<xi<60
        """
        aa = 99
        b = 25
        c = -50
        d = 1 / 1.5
        f = 0.01

        x = X[:, 0]  # (n,)
        y = X[:, 1]
        z = X[:, 2]

        listI = np.arange(1, aa + 1)  # (99,)
        zi = f * listI
        yi = b + (c * np.log(1 / zi))**d  # (99,)

        # Broadcasting: yi[None,:] - z[:,None] -> (n, 99)
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



funGulfResearch = FunGulfResearch()
