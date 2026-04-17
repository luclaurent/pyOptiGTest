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


class FunSargan(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Sargan function
        Design space: -100<xi<100
        """
        a = 0.4
        nbvar = X.shape[1]

        pxy = np.zeros(X.shape[0])
        for itN in range(nbvar):
            for itM in range(itN):
                pxy = pxy + 2 * X[:, itM] * X[:, itN]

        p = nbvar * (np.sum(X ** 2, axis=1) + a * pxy)

        if grad:
            sxy = np.sum(X, axis=1)
            dxy = sxy[:, np.newaxis] - X
            dp = nbvar * (2 * X + 2 * a * dxy)

        return (p, dp) if grad else p


funSargan = FunSargan()
