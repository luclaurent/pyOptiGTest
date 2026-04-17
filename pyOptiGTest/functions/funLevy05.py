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


class FunLevy05(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Levy 05 function
        global minimum : f(x)=-176.1375 for x=[-1.3068 -1.4248]
        Design space: -10<xi<10
        """
        m = 5
        a = 1.42513
        b = 0.80032

        x = X[:, 0]
        y = X[:, 1]

        lI = np.arange(1, m + 1)  # (5,)

        # Broadcasting: x[:,None] with lI[None,:]
        pa = np.cos((lI[np.newaxis, :] - 1) * x[:, np.newaxis] + lI[np.newaxis, :])
        pb = np.cos((lI[np.newaxis, :] + 1) * y[:, np.newaxis] + lI[np.newaxis, :])
        pc = x + a
        pd = y + b

        p = np.sum(lI[np.newaxis, :] * pa, axis=1) * np.sum(lI[np.newaxis, :] * pb, axis=1) + pc**2 + pd**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = np.sum(-lI[np.newaxis, :] * (lI[np.newaxis, :] - 1) * np.sin((lI[np.newaxis, :] - 1) * x[:, np.newaxis] + lI[np.newaxis, :]), axis=1) * np.sum(lI[np.newaxis, :] * pb, axis=1) + 2 * pc
            dp[:, 1] = np.sum(-lI[np.newaxis, :] * (lI[np.newaxis, :] + 1) * np.sin((lI[np.newaxis, :] + 1) * y[:, np.newaxis] + lI[np.newaxis, :]), axis=1) * np.sum(lI[np.newaxis, :] * pa, axis=1) + 2 * pd

        return (p, dp) if grad else p



funLevy05 = FunLevy05()
