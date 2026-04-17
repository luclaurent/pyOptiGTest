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


class FunAckley4(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Ackley's function 4 (Modified Ackley Function)
        numerous local minima
        in dimension 2: x{-1.479252, -0.739807), (1.479252, -0.739807)} >> f(x)=-3.917275
        design space -35<xi<35 (small range -2<xi<2)
        """
        a = 0.2
        b = 3
        c = 2
        nbvar = X.shape[1]

        ex1 = np.exp(-a)
        normP = np.zeros((X.shape[0], nbvar - 1))
        for itV in range(nbvar - 1):
            normP[:, itV] = np.sqrt(X[:, itV]**2 + X[:, itV + 1]**2)
        cx = np.cos(c * X[:, :-1])
        sx = np.sin(c * X[:, 1:])
        p = np.sum(ex1 * normP + b * (cx + sx), axis=1)

        if grad:
            dp = np.zeros_like(X)
            for ii in range(nbvar):
                if ii == 0:
                    dp[:, ii] = ex1 * X[:, 0] / normP[:, 0] - c * b * np.sin(c * X[:, 0])
                elif ii == nbvar - 1:
                    dp[:, ii] = ex1 * X[:, nbvar - 1] / normP[:, -1] + c * b * np.cos(c * X[:, nbvar - 1])
                else:
                    dp[:, ii] = (ex1 * X[:, ii] / normP[:, ii] - c * b * np.sin(c * X[:, ii])
                                 + ex1 * X[:, ii] / normP[:, ii] + c * b * np.cos(c * X[:, ii]))

        return (p, dp) if grad else p



funAckley4 = FunAckley4()
