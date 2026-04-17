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


class FunJennrichSampson(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Jennrich-Sampson's function
        global minimum : f(x1,x2)=124.3622 for (x1,x2)=(0.257825,0.257825)
        Design space: -1<xi<1
        """
        a = 2
        b = 10

        xxx = X[:, 0]
        yyy = X[:, 1]

        ii = np.arange(1, b + 1)  # (10,)
        # exp(xxx)^ii -> broadcasting: (n,1)^(1,10)
        ex = np.exp(xxx[:, np.newaxis])**ii[np.newaxis, :]  # (n, 10)
        ey = np.exp(yyy[:, np.newaxis])**ii[np.newaxis, :]
        se = a * ii[np.newaxis, :] - (ex + ey)

        p = np.sum((a + se)**2, axis=1)

        if grad:
            iex = ii[np.newaxis, :] * ex
            iey = ii[np.newaxis, :] * ey
            dp = np.zeros_like(X)
            dp[:, 0] = np.sum(-iex * 2 * (a + se), axis=1)
            dp[:, 1] = np.sum(-iey * 2 * (a + se), axis=1)

        return (p, dp) if grad else p



funJennrichSampson = FunJennrichSampson()
