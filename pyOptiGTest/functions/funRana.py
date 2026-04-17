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


class FunRana(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Rana function
        global minimum: f(x)=-928.5478 for xi=-500
        Design space: -500<xi<500
        """
        a = 1.0

        pa = X[:, 1:] + X[:, :-1] + a
        pb_val = X[:, 1:] - X[:, :-1] + a
        ta = np.sqrt(np.abs(pa))
        tb = np.sqrt(np.abs(pb_val))
        pc = (X[:, 1:] + a) * np.cos(tb) * np.sin(ta) + X[:, :-1] * np.cos(ta) * np.sin(tb)

        p = np.sum(np.abs(pc), axis=1)

        if grad:
            # Derivatives not fully implemented in original
            dp = np.zeros_like(X)

        return (p, dp) if grad else p


funRana = FunRana()
