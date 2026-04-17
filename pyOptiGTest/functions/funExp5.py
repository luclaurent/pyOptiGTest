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


class FunExp5(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Exp5 function
        global minimum : f(x)=0 for x=[1 10 1 5 4]
        Design space: 0<xi<20
        """
        a = 10
        b = 5
        c = 11
        d = 3
        f = 4

        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        v = X[:, 3]
        w = X[:, 4]

        listI = np.arange(1, c + 1)
        pa = np.exp(-listI[np.newaxis, :] * x[:, np.newaxis] / a)
        pb = -np.exp(-listI[np.newaxis, :] * y[:, np.newaxis] / a)
        pc = -np.exp(-listI / a)
        pd = b * np.exp(-listI)
        pe = d * np.exp(-listI[np.newaxis, :] * w[:, np.newaxis] / a)
        pf = -d * np.exp(-listI * f / a)

        pt = z[:, np.newaxis] * pa + v[:, np.newaxis] * pb + pc[np.newaxis, :] + pd[np.newaxis, :] + pe + pf[np.newaxis, :]
        p = np.sum(pt**2, axis=1)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * z * np.sum(-listI[np.newaxis, :] / a * pa * pt, axis=1)
            dp[:, 1] = 2 * v * np.sum(-listI[np.newaxis, :] / a * pb * pt, axis=1)
            dp[:, 2] = 2 * np.sum(pa * pt, axis=1)
            dp[:, 3] = 2 * np.sum(pb * pt, axis=1)
            dp[:, 4] = 2 * np.sum(-listI[np.newaxis, :] / a * pe * pt, axis=1)

        return (p, dp) if grad else p



funExp5 = FunExp5()
