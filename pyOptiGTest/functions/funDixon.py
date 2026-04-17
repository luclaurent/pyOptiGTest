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


class FunDixon(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Dixon & Price function
        Design space: -10<xi<10
        """
        nbvar = X.shape[1]

        p = (X[:, 0] - 1) ** 2
        for it in range(1, nbvar):
            p = p + (it + 1) * (2 * X[:, it] ** 2 - X[:, it - 1]) ** 2

        if grad:
            dp = np.zeros_like(X)
            for it in range(nbvar):
                if it == 0:
                    dp[:, it] = 2 * (X[:, 0] - 1) - 4 * (2 * X[:, 1] ** 2 - X[:, 0])
                elif it == nbvar - 1:
                    dp[:, it] = (it + 1) * 8 * X[:, it] * (2 * X[:, it] ** 2 - X[:, it - 1])
                else:
                    dp[:, it] = (it + 1) * 8 * X[:, it] * (2 * X[:, it] ** 2 - X[:, it - 1]) \
                        - 2 * (it + 2) * (2 * X[:, it + 1] ** 2 - X[:, it])

        return (p, dp) if grad else p


funDixon = FunDixon()
