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


class FunLangermann52(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Langermann 5 2D function
        global minimum : f(x1,x2)=-1.4 for (x1,x2)=()
        Design space:0<xi<10
        """
        ma = np.array([3, 5, 2, 1, 7], dtype=float)
        mb = np.array([5, 2, 1, 4, 9], dtype=float)
        mc = np.array([1, 2, 5, 2, 3], dtype=float)
        d = np.pi
        m = 5

        xxx = X[:, 0]
        yyy = X[:, 1]

        # Broadcasting: ma,mb,mc are (m,), xxx,yyy are (n,)
        vx = xxx[:, np.newaxis] - ma[np.newaxis, :]  # (n, m)
        vy = yyy[:, np.newaxis] - mb[np.newaxis, :]
        vxy = vx**2 + vy**2
        ex = np.exp(-1 / d * vxy)
        cx = np.cos(d * vxy)
        cc = cx * mc[np.newaxis, :]

        p = np.sum(-ex * cc, axis=1)

        if grad:
            sx = np.sin(d * vxy)
            cs = sx * mc[np.newaxis, :]
            pvx = vx * ex
            pvy = vy * ex
            dp = np.zeros_like(X)
            dp[:, 0] = np.sum(2 / d * pvx * cc + 2 * d * pvx * cs, axis=1)
            dp[:, 1] = np.sum(2 / d * pvy * cc + 2 * d * pvy * cs, axis=1)

        return (p, dp) if grad else p



funLangermann52 = FunLangermann52()
