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


class FunDeVilliersGlasser2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        De Villiers-Glasser 2 function
        global minimum: f(x)=0
        Design space: 1<xi<60
        """
        a = 16
        b = 0.1
        c = 1.0
        d_val = 53.81
        f = 1.27
        g = 3.012
        h = 2.13
        k = 0.507

        ii = np.arange(1, a + 1, dtype=float)
        t = b * (ii - c)
        y = d_val * f ** t * np.tanh(g * t + np.sin(h * t)) * np.cos(np.exp(k) * t)

        xxx = X[:, 0]
        yyy = X[:, 1]
        zzz = X[:, 2]
        vvv = X[:, 3]
        www = X[:, 4]

        zt = zzz[:, np.newaxis] * t[np.newaxis, :]
        vt = vvv[:, np.newaxis] * t[np.newaxis, :]
        wt = np.exp(www)[:, np.newaxis] * t[np.newaxis, :]

        svt = np.sin(vt)
        yt = yyy[:, np.newaxis] ** t[np.newaxis, :]
        xyt = xxx[:, np.newaxis] * yt

        pa = np.tanh(zt + svt)
        pb_cos = np.cos(wt)
        pc = xyt * pa * pb_cos
        pd = pc - y[np.newaxis, :]

        p = np.sum(pd ** 2, axis=1)

        if grad:
            xt = xxx[:, np.newaxis] * t[np.newaxis, :]
            ytm = yyy[:, np.newaxis] ** (t[np.newaxis, :] - 1)
            cvt = np.cos(vt)
            pe = -np.sin(wt)

            dp = np.zeros_like(X)
            dp[:, 0] = np.sum(2 * yt * pa * pb_cos * pd, axis=1)
            dp[:, 1] = np.sum(2 * xt * ytm * pa * pb_cos * pd, axis=1)
            dp[:, 2] = np.sum(2 * xyt * t[np.newaxis, :] * (1 - pa ** 2) * pb_cos * pd, axis=1)
            dp[:, 3] = np.sum(2 * xyt * t[np.newaxis, :] * cvt * (1 - pa ** 2) * pb_cos * pd, axis=1)
            dp[:, 4] = np.sum(2 * xyt * wt * pe * pa * pd, axis=1)

        return (p, dp) if grad else p


funDeVilliersGlasser2 = FunDeVilliersGlasser2()
