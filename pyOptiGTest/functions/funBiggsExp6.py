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


class FunBiggsExp6(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Biggs EXP 6's function
        1 global minimum : x=(1, 10, 1, 5, 4, 3) >> f(x)=0
        design space 0<xi<20
        """
        a = 5
        b = 10
        c = 3
        d = 4
        nbt = 11
        t = 0.1 * np.arange(1, nbt + 1)
        y = np.exp(-t) - a * np.exp(-b * t) + c * np.exp(-d * t)

        xxx = X[:, 0]
        yyy = X[:, 1]
        zzz = X[:, 2]
        lll = X[:, 3]
        mmm = X[:, 4]
        nnn = X[:, 5]

        p = np.zeros_like(xxx)
        for it in range(nbt):
            p = p + (zzz * np.exp(-t[it] * xxx) - lll * np.exp(-t[it] * yyy) + nnn * np.exp(-t[it] * mmm) - y[it])**2

        if grad:
            dp = np.zeros((xxx.shape[0], 6))
            for it in range(nbt):
                mult = zzz * np.exp(-t[it] * xxx) - lll * np.exp(-t[it] * yyy) + nnn * np.exp(-t[it] * mmm) - y[it]
                dp[:, 0] += -2 * zzz * t[it] * np.exp(-t[it] * xxx) * mult
                dp[:, 1] += 2 * lll * t[it] * np.exp(-t[it] * yyy) * mult
                dp[:, 2] += 2 * np.exp(-t[it] * xxx) * mult
                dp[:, 3] += -2 * np.exp(-t[it] * yyy) * mult
                dp[:, 4] += -2 * nnn * t[it] * np.exp(-t[it] * mmm) * mult
                dp[:, 5] += 2 * np.exp(-t[it] * mmm) * mult

        return (p, dp) if grad else p



funBiggsExp6 = FunBiggsExp6()
