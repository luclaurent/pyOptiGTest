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


class FunZimmerman(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Zimmerman function
        1 minimum global: f(7,2)=0
        Design space 0<xi<100
        """
        a = 9
        b = 3
        c = 2
        d = 14
        e = 100

        x = X[:, 0]
        y = X[:, 1]

        Zh1 = a - x - y
        Zh2 = (x - b)**2 + (y - c)**2
        Zh3 = x * y - d
        Zp = lambda t: e * (a + t)

        pA = np.zeros((X.shape[0], 5))
        pA[:, 0] = Zh1
        pA[:, 1] = Zp(Zh2) * np.sign(Zh2)
        pA[:, 2] = Zp(Zh3) * np.sign(Zh3)
        pA[:, 3] = Zp(-x) * np.sign(x)
        pA[:, 4] = Zp(-y) * np.sign(y)

        p = np.max(pA, axis=1)

        if grad:
            dp = np.zeros_like(X)

        return (p, dp) if grad else p



funZimmerman = FunZimmerman()
