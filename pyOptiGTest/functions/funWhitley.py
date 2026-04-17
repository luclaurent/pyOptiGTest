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


class FunWhitley(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Whitley function
        1 minimum global: f(1,...,1)=0
        Design space -10.24<xi<10.24
        """
        a = 100
        b = 1
        c = 4000

        n = X.shape[0]
        d = X.shape[1]

        # xxi^2 - xxj for all pairs (i,j)
        # X[:, :, None] is (n, d, 1), X[:, None, :] is (n, 1, d)
        xxi = X[:, :, np.newaxis]  # (n, d, 1) - will broadcast as i index
        xxj = X[:, np.newaxis, :]  # (n, 1, d) - will broadcast as j index

        xxij = xxi**2 - xxj  # (n, d, d) - xxij[s,i,j] = X[s,i]^2 - X[s,j]
        pA = a * xxij**2 + (b - xxj)**2
        pAA = pA**2 / c
        pB = np.cos(pA)

        # sum over j (axis=2) then over i (axis=1)
        p = np.sum(np.sum(pAA - pB + b, axis=2), axis=1)

        if grad:
            # dgX: derivative through xxi (the i-th variable)
            dgX = a * 8 / c * xxi * xxij * (pA + np.sin(pA))
            # dgY: derivative through xxj (the j-th variable)
            dgY = (-2 * a / c * xxij - 2 / c * (b - xxj)) * pA - (2 * a * xxij + 2 * (b - xxj)) * np.sin(pA)
            # Sum dgX over j (axis=2) for contribution to dp[:,i]
            # Sum dgY over i (axis=1) for contribution to dp[:,j]
            dp = np.sum(dgX, axis=2) + np.sum(dgY, axis=1)

        return (p, dp) if grad else p



funWhitley = FunWhitley()
