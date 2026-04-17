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


class FunCorana(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Corana's function
        global minimum : f(x1,x2,x3,x4)=0 for (0,0,0,0)
        Design space: -500<xi<500
        """
        a = 0.05
        b = 0.2
        e = 0.49999
        d = np.array([1.0, 1e3, 10.0, 1e2])

        n_vars = X.shape[1]
        # d repeats cyclically if more than 4 vars, or truncates
        dr = np.tile(d, (n_vars // 4) + 1)[:n_vars]

        zz = b * np.floor(np.abs(X) / b + e) * np.sign(X)
        vv = np.abs(X - zz)
        maskV = vv < a

        pz = zz - a * np.sign(zz)
        pd = dr[np.newaxis, :] * X

        pS = np.zeros_like(X)
        pS[maskV] = pz[maskV]
        pS[~maskV] = pd[~maskV]

        p = np.sum(pS, axis=1)

        if grad:
            dp = np.zeros_like(X)

        return (p, dp) if grad else p



funCorana = FunCorana()
