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


class FunStretchedV(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Stretched V function
        1 global minimum: f(0,...,0)=0
        Design space: -10<xi<10
        """
        a = 0.25
        b = 50.0
        c = 0.1
        d = 1.0
        nbvar = X.shape[1]

        xkp = X[:, 1:]
        xk = X[:, :-1]
        gval = xkp ** 2 + xk ** 2
        pa = gval ** a * (np.sin(b * gval ** c) + d) ** 2

        p = np.sum(pa, axis=1)

        if grad:
            dp = np.zeros_like(X)
            for itX in range(nbvar):
                if itX == 0:
                    x_val = X[:, 1]
                    y_val = X[:, 0]
                    g = x_val ** 2 + y_val ** 2
                    dg = 2 * y_val
                    sinterm = np.sin(b * g ** c) + d
                    dp[:, itX] = (a * dg * g ** (a - 1) * sinterm ** 2
                                  + 2 * g ** a * c * b * dg * g ** (c - 1) * np.cos(b * g ** c) * sinterm)
                elif itX == nbvar - 1:
                    x_val = X[:, nbvar - 1]
                    y_val = X[:, nbvar - 2]
                    g = x_val ** 2 + y_val ** 2
                    dg = 2 * x_val
                    sinterm = np.sin(b * g ** c) + d
                    dp[:, itX] = (a * dg * g ** (a - 1) * sinterm ** 2
                                  + 2 * g ** a * c * b * dg * g ** (c - 1) * np.cos(b * g ** c) * sinterm)
                else:
                    # Pair (itX, itX-1)
                    g1 = X[:, itX] ** 2 + X[:, itX - 1] ** 2
                    dg1 = 2 * X[:, itX]
                    s1 = np.sin(b * g1 ** c) + d
                    term1 = (a * dg1 * g1 ** (a - 1) * s1 ** 2
                             + 2 * g1 ** a * c * b * dg1 * g1 ** (c - 1) * np.cos(b * g1 ** c) * s1)
                    # Pair (itX+1, itX)
                    g2 = X[:, itX + 1] ** 2 + X[:, itX] ** 2
                    dg2 = 2 * X[:, itX]
                    s2 = np.sin(b * g2 ** c) + d
                    term2 = (a * dg2 * g2 ** (a - 1) * s2 ** 2
                             + 2 * g2 ** a * c * b * dg2 * g2 ** (c - 1) * np.cos(b * g2 ** c) * s2)
                    dp[:, itX] = term1 + term2

        return (p, dp) if grad else p


funStretchedV = FunStretchedV()
