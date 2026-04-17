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


class FunWeierstrass(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Weierstrass function
        1 global minimum: f(0,...,0)=0
        Design space: -0.5<xi<0.5
        """
        a = 0.5
        b = 3.0
        k_max = 20
        c = 2 * np.pi
        d = 0.5
        e = np.pi
        nbvar = X.shape[1]

        lik = np.arange(0, k_max + 1, dtype=float)  # 0..20

        # For each variable and each k: a^k * cos(c * b^k * (x + d))
        # X shape: (n, nbvar), lik shape: (k+1,)
        # Broadcasting: (n, nbvar, 1) and (1, 1, k+1)
        ak = a ** lik  # (k+1,)
        bk = b ** lik  # (k+1,)

        # pa[i,j,l] = a^l * cos(c * b^l * (X[i,j] + d))
        arg = c * bk[np.newaxis, np.newaxis, :] * (X[:, :, np.newaxis] + d)
        pa = ak[np.newaxis, np.newaxis, :] * np.cos(arg)

        # pb[l] = a^l * cos(pi * b^l)
        pb = ak * np.cos(e * bk)

        pAA = np.sum(pa, axis=2)  # (n, nbvar)
        pBB = np.sum(pb)  # scalar

        p = np.sum(pAA, axis=1) - nbvar * pBB

        if grad:
            dp = np.sum(-ak[np.newaxis, np.newaxis, :] * bk[np.newaxis, np.newaxis, :] * c
                         * np.sin(arg), axis=2)

        return (p, dp) if grad else p


funWeierstrass = FunWeierstrass()
