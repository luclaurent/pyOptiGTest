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


class FunObjKursawe2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Kursawe 2nd objective function.
        f2(x) = sum_{i=1}^{n} [|x_i|^0.8 + 5*sin(x_i^3)]
        Design space: -5 <= xi <= 5, dim=3
        """

        a = 0.8
        b = 5
        c = 3
        p = np.sum(np.abs(X)**a + b * np.sin(X**c), axis=1)

        if grad:
            dp = a * np.abs(X)**(a - 1) * np.sign(X) + b * c * X**(c - 1) * np.cos(X**c)
            return p, dp
        return p


funObjKursawe2 = FunObjKursawe2()
