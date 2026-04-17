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


class FunPenalty2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Penalty 2 function
        global minimum: f(x)=0 for xi=1
        Design space: -50<xi<50
        """
        a = 0.1
        b = 3 * np.pi
        c = 1.0
        d = 2 * np.pi
        e = 5.0
        f = 100.0
        g = 4.0

        pa = np.sin(b * X[:, 0])
        pb = X[:, :-1] - c
        sc = np.sin(b * X[:, 1:])
        pc = c + sc ** 2
        pd = X[:, -1] - c
        se = np.sin(d * X[:, -1])
        pe = c + se ** 2

        u = np.zeros_like(X)
        IXb = np.abs(X) > e
        u[IXb] = f * (np.abs(X[IXb]) - e) ** g

        p = a * (pa ** 2 + np.sum(pb ** 2 * pc, axis=1) + pd ** 2 * pe) + np.sum(u, axis=1)

        if grad:
            dp = np.zeros_like(X)
            du = np.zeros_like(X)
            du[IXb] = f * e * np.sign(X[IXb]) * (np.abs(X[IXb]) - b) ** (f - 1)

            ppa = np.cos(b * X[:, 0])
            cc = np.cos(b * X[:, 1:])
            ce = np.cos(d * X[:, -1])

            dp[:, 0] = (2 * a * b * pa * ppa
                         + 2 * a * pb[:, 0] * pc[:, 0]
                         + du[:, 0])
            dp[:, -1] = (2 * a * b * sc[:, -1] * cc[:, -1] * pb[:, -1] ** 2
                          + 2 * a * pd * pe
                          + 2 * d * a * ce * se * pd ** 2
                          + du[:, -1])
            if X.shape[1] > 2:
                dp[:, 1:-1] = (2 * a * b * sc[:, :-1] * cc[:, :-1] * pb[:, :-1] ** 2
                               + 2 * a * pb[:, 1:] * pc[:, 1:]
                               + du[:, 1:-1])

        return (p, dp) if grad else p


funPenalty2 = FunPenalty2()
