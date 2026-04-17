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


class FunUrsem4(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Ursem 4 function
        1 minimum global: f(0,0)=-1.5
        Design space -2<xi<2
        """
        a = 3
        b = np.pi / 2
        c = np.pi / 2
        d = 2
        e = 4

        x = X[:, 0]
        y = X[:, 1]

        sxy = np.sqrt(x**2 + y**2)

        pa = -a * np.sin(b * x + c)
        pb = (d - sxy) / e

        p = pa * pb

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = -a * b * np.cos(b * x + c) * pb - pa * x / (e * (sxy + 1e-300))
            dp[:, 1] = -pa * y / (e * (sxy + 1e-300))

        return (p, dp) if grad else p



funUrsem4 = FunUrsem4()
