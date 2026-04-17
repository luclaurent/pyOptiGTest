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


class FunMishra08(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Mishra 8 function
        Design space: -10<xi<10
        """
        aa = 0.001
        a_coefs = np.array([1, -20, 180, -960, 3360, -8064, 13340, -15360, 11520, -5120, 2624], dtype=float)
        b_coefs = np.array([1, 12, 54, 108, 81], dtype=float)

        xxx = X[:, 0]
        yyy = X[:, 1]

        # Evaluate polynomial g(x) = sum(a_i * x^(10-i)) for i=0..10
        g = np.zeros_like(xxx)
        for i in range(11):
            g = g + a_coefs[i] * xxx ** (10 - i)

        # Evaluate polynomial h(y) = sum(b_i * y^(4-i)) for i=0..4
        h = np.zeros_like(yyy)
        for i in range(5):
            h = h + b_coefs[i] * yyy ** (4 - i)

        pa = np.abs(h) * np.abs(g)
        p = aa * pa ** 2

        if grad:
            # dg/dx
            dg = np.zeros_like(xxx)
            for i in range(10):
                dg = dg + a_coefs[i] * (10 - i) * xxx ** (9 - i)
            # dh/dy
            dh = np.zeros_like(yyy)
            for i in range(4):
                dh = dh + b_coefs[i] * (4 - i) * yyy ** (3 - i)

            dp = np.zeros_like(X)
            dp[:, 0] = 2 * aa * np.sign(g) * np.abs(h) * dg * pa
            dp[:, 1] = 2 * aa * np.sign(h) * np.abs(g) * dh * pa

        return (p, dp) if grad else p


funMishra08 = FunMishra08()
