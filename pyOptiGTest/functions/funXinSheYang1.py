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


class FunXinSheYang1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Xin-She Yang 1 function
        1 global minimum: f(0,...,0)=0
        Design space: -5<xi<5
        """
        nbvar = X.shape[1]
        ee = np.random.rand(*X.shape)
        li = np.arange(1, nbvar + 1, dtype=float)[np.newaxis, :]

        pA = ee * np.abs(X) ** li
        p = np.sum(pA, axis=1)

        if grad:
            dp = ee * li * np.sign(X) * np.abs(X) ** (li - 1)

        return (p, dp) if grad else p


funXinSheYang1 = FunXinSheYang1()
