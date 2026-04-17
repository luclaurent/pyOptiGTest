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


class FunObjFonsecaFleming2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Fonseca and Fleming 2nd objective function.
        f2(x) = 1 - exp(-sum((xi + 1/sqrt(n))^2))
        Design space: -4 <= xi <= 4
        """

        n = X.shape[1]
        a = 1.0 / np.sqrt(n)
        d = X + a
        s = np.sum(d**2, axis=1)
        p = 1 - np.exp(-s)

        if grad:
            dp = 2 * d * np.exp(-s)[:, np.newaxis]
            return p, dp
        return p


funObjFonsecaFleming2 = FunObjFonsecaFleming2()
