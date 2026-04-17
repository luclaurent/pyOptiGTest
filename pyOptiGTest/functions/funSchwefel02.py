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


class FunSchwefel02(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Schwefel 2 function
        1 global minimum: f(0,...,0)=0
        Design space: -100<xi<100
        """
        nbvar = X.shape[1]
        n_samples = X.shape[0]

        p = np.zeros(n_samples)
        for itX in range(nbvar):
            p = p + np.sum(X[:, :itX + 1], axis=1) ** 2

        if grad:
            dp = np.zeros_like(X)
            for itX in range(nbvar):
                for itD in range(itX, nbvar):
                    dp[:, itX] = dp[:, itX] + 2 * np.sum(X[:, :itD + 1], axis=1)

        return (p, dp) if grad else p


funSchwefel02 = FunSchwefel02()
