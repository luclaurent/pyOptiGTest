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


class FunMichalewicz(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Michalewicz function
        global minimum: depends on the number of variables
        Design space: 0<xi<pi
        """
        nbvar = X.shape[1]

        p = np.zeros(X.shape[0])
        for it in range(1, nbvar + 1):
            p = p + np.sin(X[:, it - 1]) * np.sin(it * X[:, it - 1]**2 / np.pi)**20
        p = -p

        if grad:
            dp = np.zeros_like(X)
            for it in range(1, nbvar + 1):
                dp[:, it - 1] = (np.cos(X[:, it - 1]) * np.sin(it * X[:, it - 1]**2 / np.pi)**20
                                 + 40 * it / np.pi * X[:, it - 1] * np.sin(X[:, it - 1])
                                 * np.cos(it * X[:, it - 1]**2 / np.pi)**19)
            dp = -dp

        return (p, dp) if grad else p



funMichalewicz = FunMichalewicz()
