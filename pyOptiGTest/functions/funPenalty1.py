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


class FunPenalty1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Penalty 1 function
        global minimum: f(x)=0 for xi=-1
        Design space: -50<xi<50
        """
        a_coef = np.pi / 30
        b = 10.0
        c = np.pi
        d = 1.0
        e = 100.0
        f = 4.0

        y = d + 1.0 / f * (X + d)

        pa = np.sin(c * y[:, 0])
        pb = y[:, :-1] - d
        sc = np.sin(c * y[:, 1:])
        pc = d + b * sc ** 2
        pd = y[:, -1] - d

        u = np.zeros_like(X)
        IXb = np.abs(X) > b
        u[IXb] = e * (np.abs(X[IXb]) - b) ** f

        p = a_coef * (b * pa ** 2 + np.sum(pb ** 2 * pc, axis=1) + pd ** 2) + np.sum(u, axis=1)

        if grad:
            dp = np.zeros_like(X)
            du = np.zeros_like(X)
            du[IXb] = f * e * np.sign(X[IXb]) * (np.abs(X[IXb]) - b) ** (f - 1)

            ppa = np.cos(c * y[:, 0])
            cc = np.cos(c * y[:, 1:])

            dp[:, 0] = (2 * a_coef * b * c / f * pa * ppa
                         + 2 * a_coef / f * pb[:, 0] * pc[:, 0]
                         + du[:, 0])
            dp[:, -1] = (2 * a_coef * b * c / f * sc[:, -1] * cc[:, -1] * pb[:, -1] ** 2
                          + 2 * a_coef / f * pd
                          + du[:, -1])
            if X.shape[1] > 2:
                dp[:, 1:-1] = (2 * a_coef * b * c / f * sc[:, :-1] * cc[:, :-1] * pb[:, :-1] ** 2
                               + 2 * a_coef / f * pb[:, 1:] * pc[:, 1:]
                               + du[:, 1:-1])

        return (p, dp) if grad else p


funPenalty1 = FunPenalty1()
