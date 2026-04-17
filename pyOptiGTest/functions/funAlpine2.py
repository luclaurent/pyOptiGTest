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


class FunAlpine2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Alpine 2 function
        1 global minimum : x=(7.917,...,7.917) >> f(x)=2.808^n
        Design space: 0<xi<10
        """
        nbvar = X.shape[1]

        fx = np.sqrt(X) * np.sin(X)
        p = np.prod(fx, axis=1)

        if grad:
            dp = np.zeros_like(X)
            for itV in range(nbvar):
                others = list(range(nbvar))
                others.remove(itV)
                dp[:, itV] = np.prod(fx[:, others], axis=1) * \
                    (0.5 / np.sqrt(X[:, itV]) * np.sin(X[:, itV]) + np.sqrt(X[:, itV]) * np.cos(X[:, itV]))

        return (p, dp) if grad else p


funAlpine2 = FunAlpine2()
