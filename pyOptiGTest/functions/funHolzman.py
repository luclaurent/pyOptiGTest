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


class FunHolzman(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Holzman function
        global minimum: f(x)=0 for x=[50, 25, 1.5]
        Design space: 0<x1<100, 0<x2<25.6, 0<x3<5
        """
        a = 0.01
        b_val = 1.0
        c = 25.0
        d = 50.0
        e = 2.0 / 3.0
        f = 1e-2
        m = 99

        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]

        listI = np.arange(0, m + 1, dtype=float)  # 0..99
        u = c + (-d * np.log(f * (listI + 1))) ** e  # shape (m+1,)

        # Broadcasting: (n,1) op (1,m+1) -> (n, m+1)
        pa = u[np.newaxis, :] - y[:, np.newaxis]
        pax = pa ** z[:, np.newaxis]
        paxx = pax / x[:, np.newaxis]
        pb_val = np.exp(-paxx) - a * (listI[np.newaxis, :] + b_val)

        p = np.sum(pb_val, axis=1)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0] = np.sum(paxx / x[:, np.newaxis] * np.exp(-paxx), axis=1)
            dp[:, 1] = np.sum(z[:, np.newaxis] * pa ** (z[:, np.newaxis] - 1) / x[:, np.newaxis] * np.exp(-paxx), axis=1)
            dp[:, 2] = np.sum(-np.log(np.maximum(pa, 1e-300)) * paxx * np.exp(-paxx), axis=1)

        return (p, dp) if grad else p


funHolzman = FunHolzman()
