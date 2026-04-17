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


class FunDixonPrice(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Dixon-Price function
        1 global minimum
        Design space: -10<xi<10
        """
        dim = X.shape[1]
        a = 1.0
        b = 2.0

        pa = (X[:, 0] - a) ** 2
        ii = np.arange(2, dim + 1, dtype=float)  # 2..dim
        pb = b * X[:, 1:] ** 2 - X[:, :-1]
        pc = pb ** 2 * ii[np.newaxis, :]
        p = pa + np.sum(pc, axis=1)

        if grad:
            dp = np.zeros_like(X)
            pd = pb * ii[np.newaxis, :]
            dp[:, 0] = 2 * (X[:, 0] - a) - 2 * pd[:, 0]
            for it in range(1, dim - 1):
                try:
                    dp[:, it] = 4 * b * (it + 1) * X[:, it] * pb[:, it - 1] \
                        - 2 * (it + 2) * pb[:, it]
                except Exception:
                    pass
            dp[:, dim - 1] = 4 * b * dim * X[:, -1] * pb[:, -1]

        return (p, dp) if grad else p


funDixonPrice = FunDixonPrice()
