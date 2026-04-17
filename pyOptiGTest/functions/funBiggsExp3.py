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


class FunBiggsExp3(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Biggs EXP 3's function
        1 global minimum : x=(1, 10, 5) >> f(x)=0
        design space 0<xi<20
        """
        a = 5
        b = 10
        nbt = 10
        t = 0.1 * np.arange(1, 11)
        y = np.exp(-t) - a * np.exp(-b * t)

        xxx = X[:, 0]
        yyy = X[:, 1]
        zzz = X[:, 2]

        p = np.zeros_like(xxx)
        for it in range(nbt):
            p = p + (np.exp(-t[it] * xxx) - zzz * np.exp(-t[it] * yyy) - y[it])**2

        if grad:
            dp = np.zeros((xxx.shape[0], 3))
            for it in range(nbt):
                res = np.exp(-t[it] * xxx) - zzz * np.exp(-t[it] * yyy) - y[it]
                dp[:, 0] += -2 * t[it] * np.exp(-t[it] * xxx) * res
                dp[:, 1] += 2 * zzz * t[it] * np.exp(-t[it] * yyy) * res
                dp[:, 2] += -2 * np.exp(-t[it] * yyy) * res

        return (p, dp) if grad else p



funBiggsExp3 = FunBiggsExp3()
