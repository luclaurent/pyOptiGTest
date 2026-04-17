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


class FunXinSheYang3(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Xin-She-Yang 3 function
        1 minimum global: f(0,...,0)=-1
        Design space -20<xi<20
        """
        a = 2
        m = 3
        beta = 15

        pa = X / beta
        pA = np.sum(pa**(2 * m), axis=1)
        pAA = np.exp(-pA)

        pb = X**2
        pB = np.sum(pb, axis=1)
        pBB = np.exp(-pB)

        pc = np.cos(X)**2
        pCC = np.prod(pc, axis=1)

        p = pAA - a * pBB * pCC

        if grad:
            dp = (-2 * m / beta**(2 * m) * X**(2 * m - 1) * pAA[:, np.newaxis]
                  + 2 * a * X * pBB[:, np.newaxis] * pCC[:, np.newaxis]
                  + 2 * a * np.sin(X) * np.cos(X) * pBB[:, np.newaxis] * pCC[:, np.newaxis] / pc)

        return (p, dp) if grad else p



funXinSheYang3 = FunXinSheYang3()
