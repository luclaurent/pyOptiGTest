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


class FunBrad(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bard's function
        1 global minimum : x=(0.0824, 1.133, 2.3437) >> f(x)=0.00821487
        Design space: -0.25<x1<0.25, 0.01<x2,x3<2.5
        """
        nbt = 15
        u = np.arange(1, nbt + 1, dtype=float)
        v = 16.0 - u
        w = np.minimum(u, v)
        y = np.array([0.14, 0.18, 0.22, 0.25, 0.29, 0.32, 0.35, 0.39,
                      0.37, 0.58, 0.73, 0.96, 1.34, 2.10, 4.39])

        xxx = X[:, 0]
        yyy = X[:, 1]
        zzz = X[:, 2]

        p = np.zeros_like(xxx)
        for it in range(nbt):
            p = p + ((y[it] - xxx - u[it]) / (v[it] * yyy + w[it] * zzz)) ** 2

        if grad:
            dp = np.zeros((xxx.shape[0], 3))
            for it in range(nbt):
                denom = v[it] * yyy + w[it] * zzz
                mult = (y[it] - xxx - u[it]) / denom
                dp[:, 0] = dp[:, 0] - 2.0 / denom * mult
                dp[:, 1] = dp[:, 1] - (y[it] - xxx - u[it]) * v[it] / denom ** 2 * mult
                dp[:, 2] = dp[:, 2] - (y[it] - xxx - u[it]) * w[it] / denom ** 2 * mult

        return (p, dp) if grad else p


funBrad = FunBrad()
