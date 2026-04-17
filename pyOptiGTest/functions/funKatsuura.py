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


class FunKatsuura(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Katsuura function
        Design space: 0<xi<100
        """
        d = 32
        nbvar = X.shape[1]
        n_samples = X.shape[0]

        # lD: powers of 2 from 2^1 to 2^32, shape (d,)
        lD = 2.0 ** np.arange(1, d + 1, dtype=float)

        p = np.ones(n_samples)
        for i in range(nbvar):
            xi = X[:, i]  # (n,)
            pa = lD[np.newaxis, :] * xi[:, np.newaxis]  # (n, d)
            pbf = np.floor(pa) * (1.0 / lD[np.newaxis, :])
            spb = np.sum(pbf, axis=1)
            pt = 1 + (i + 1) * spb
            p = p * pt

        if grad:
            dp = np.zeros_like(X)
            is_int = (np.round(X) == X)
            dp[is_int] = 1.0 / np.finfo(float).eps

        return (p, dp) if grad else p


funKatsuura = FunKatsuura()
