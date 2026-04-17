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


class FunKowalik(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Kowalik function
        global minimum : f(x)=0.00030748610 for x=[0.192833 0.190836 0.123117 0.135766]
        Design space: -5<xi<5
        """
        m = 11
        a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
        b = np.array([4, 2, 1, 1/2, 1/4, 1/6, 1/8, 1/10, 1/12, 1/14, 1/16])

        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        w = X[:, 3]

        # Broadcasting: a,b are (m,), x,y,z,w are (n,)
        # pa = b^2 + b*y -> (n, m)
        pa = b[np.newaxis, :]**2 + b[np.newaxis, :] * y[:, np.newaxis]
        pb = b[np.newaxis, :]**2 + b[np.newaxis, :] * z[:, np.newaxis] + w[:, np.newaxis]

        pt = a[np.newaxis, :] - x[:, np.newaxis] * pa / pb
        p = np.sum(pt**2, axis=1)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = np.sum(-2 * pa / pb * pt, axis=1)
            dp[:, 1] = np.sum(-2 * b[np.newaxis, :] * x[:, np.newaxis] / pb * pt, axis=1)
            dp[:, 2] = np.sum(2 * b[np.newaxis, :] * x[:, np.newaxis] * pa / pb**2 * pt, axis=1)
            dp[:, 3] = np.sum(2 * x[:, np.newaxis] * pa / pb**2 * pt, axis=1)

        return (p, dp) if grad else p



funKowalik = FunKowalik()
