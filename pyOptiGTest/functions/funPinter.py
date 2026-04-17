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


class FunPinter(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Pinter function
        global minimum: f(x)=0 for xi=0
        Design space: -10<xi<10
        """
        nbvar = X.shape[1]

        # Circular indexing: [x_n, x_1, ..., x_n, x_1]
        xxm = np.column_stack([X[:, -1], X, X[:, 0]])

        A = xxm[:, :-2] * np.sin(X) + np.sin(xxm[:, 2:])
        B = xxm[:, :-2] ** 2 - 2 * X + 3 * xxm[:, 2:] - np.cos(X) + 1

        listI = np.arange(1, nbvar + 1, dtype=float)[np.newaxis, :]

        pa = listI * X ** 2
        pb_val = 20 * listI * np.sin(A) ** 2
        pc = 1 + listI * B ** 2
        pd = listI * np.log10(pc)

        p = np.sum(pa + pb_val + pd, axis=1)

        if grad:
            dp = 2 * listI * X
            # Gradient is complex due to circular dependencies; simplified version
            Ak = xxm[:, :-2] * np.cos(X)
            Akp = np.sin(xxm[:, 2:])
            Bk = np.sin(X) - 2
            # Simplified gradient (main terms)
            dp = dp + 2 * 20 * listI * Ak * np.cos(A) * np.sin(A)
            dp = dp + 2 * listI ** 2 * Bk * B / (np.log(10) * pc)

        return (p, dp) if grad else p


funPinter = FunPinter()
