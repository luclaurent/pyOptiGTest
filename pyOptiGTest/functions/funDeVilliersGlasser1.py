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


class FunDeVilliersGlasser1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        De Villiers-Glasser 1 function
        global minimum: f(x)=0
        Design space: 1<xi<100
        """
        a = 24
        b = 0.1
        c = 1.0
        d_val = 60.137
        f = 1.371
        g = 3.112
        h = 1.761

        ii = np.arange(1, a + 1, dtype=float)
        t = b * (ii - c)  # shape (a,)
        y = d_val * f ** t * np.sin(g * t + h)  # shape (a,)

        xxx = X[:, 0]
        yyy = X[:, 1]
        zzz = X[:, 2]
        vvv = X[:, 3]

        # Broadcast: xxx is (n,), t is (a,) -> (n, a)
        zt = zzz[:, np.newaxis] * t[np.newaxis, :]
        ztv = zt + vvv[:, np.newaxis]
        szv = np.sin(ztv)

        yt = yyy[:, np.newaxis] ** t[np.newaxis, :]
        xyt = xxx[:, np.newaxis] * yt

        pa = szv * xyt
        pb = pa - y[np.newaxis, :]

        p = np.sum(pb ** 2, axis=1)

        if grad:
            czv = np.cos(ztv)
            xt = xxx[:, np.newaxis] * t[np.newaxis, :]
            ytm = yyy[:, np.newaxis] ** (t[np.newaxis, :] - 1)

            dp = np.zeros_like(X)
            dp[:, 0] = np.sum(2 * yt * szv * pb, axis=1)
            dp[:, 1] = np.sum(2 * xt * ytm * szv * pb, axis=1)
            dp[:, 2] = np.sum(2 * xyt * t[np.newaxis, :] * czv * pb, axis=1)
            dp[:, 3] = np.sum(2 * xyt * czv * pb, axis=1)

        return (p, dp) if grad else p


funDeVilliersGlasser1 = FunDeVilliersGlasser1()
