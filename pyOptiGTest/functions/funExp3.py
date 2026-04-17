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


class FunExp3(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Exp3 function
        global minimum : f(x)=0 for x=[1 10 5]
        Design space: 0<xi<20
        """
        a = 10
        b = 5

        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]

        listI = np.arange(1, a + 1)
        pa = np.exp(-listI[np.newaxis, :] * x[:, np.newaxis] / a)
        pb = -np.exp(-listI[np.newaxis, :] * y[:, np.newaxis] / a)
        pc = -np.exp(-listI / a)
        pd = b * np.exp(-listI)

        pt = pa + z[:, np.newaxis] * pb + pc[np.newaxis, :] + pd[np.newaxis, :]
        p = np.sum(pt**2, axis=1)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = 2 * np.sum(-listI[np.newaxis, :] / a * pa * pt, axis=1)
            dp[:, 1] = 2 * z * np.sum(-listI[np.newaxis, :] / a * pb * pt, axis=1)
            dp[:, 2] = 2 * np.sum(pb * pt, axis=1)

        return (p, dp) if grad else p



funExp3 = FunExp3()
