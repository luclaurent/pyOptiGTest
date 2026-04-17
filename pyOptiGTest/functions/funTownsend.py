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


class FunTownsend(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Townsend function (objective function for constrained problem).
        L. LAURENT -- 02/05/2018 -- luc.laurent@lecnam.net
        global minimum : f(x1,x2)=-2.0239884 at (2.0052938, 1.1944506)
        Design space: -2.25 < x1 < 2.5, -2.5 < x2 < 1.75
        """

        xxx = X[:, 0]
        yyy = X[:, 1]

        a = 0.1
        b = 3.0

        cxy = np.cos((xxx - a) * yyy)
        sxy = np.sin(b * xxx + yyy)
        p = -cxy**2 - xxx * sxy

        if grad:
            dp = np.zeros_like(X)
            sxya = np.sin((xxx - a) * yyy)
            dp[:, 0] = 2 * yyy * sxya * cxy - sxy - b * xxx * np.cos(b * xxx + yyy)
            dp[:, 1] = 2 * (xxx - a) * sxya * cxy - xxx * np.cos(b * xxx + yyy)
            return p, dp
        return p


funTownsend = FunTownsend()
