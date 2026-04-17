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


class FunHartmann6(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Hartmann 6 function
        global minimum: f(x)=-3.32237
        Design space: 0<xi<1
        """
        ma = np.array([[10.00, 0.05, 3.00, 17.00],
                       [3.00, 10.00, 3.50, 8.00],
                       [17.00, 17.00, 1.70, 0.05],
                       [3.50, 0.10, 10.00, 10.00],
                       [1.70, 8.00, 17.00, 0.10],
                       [8.00, 14.00, 8.00, 14.00]])
        mp = np.array([[0.1312, 0.2329, 0.2348, 0.4047],
                       [0.1696, 0.4135, 0.1451, 0.8828],
                       [0.5569, 0.8307, 0.3522, 0.8732],
                       [0.0124, 0.3736, 0.2883, 0.5743],
                       [0.8283, 0.1004, 0.3047, 0.1091],
                       [0.5886, 0.9991, 0.6650, 0.0381]])
        c = np.array([1.0, 1.2, 3.0, 3.2])
        b = 4
        dim = 6

        XX = X[:, :dim]
        n_samples = XX.shape[0]
        p = np.zeros(n_samples)
        for it in range(b):
            pa = XX - mp[:dim, it][np.newaxis, :]
            pb_val = ma[:dim, it][np.newaxis, :] * pa**2
            p = p - c[it] * np.exp(-np.sum(pb_val, axis=1))

        if grad:
            dp = np.zeros_like(X)
            for it in range(b):
                pa = XX - mp[:dim, it][np.newaxis, :]
                pb_val = ma[:dim, it][np.newaxis, :] * pa**2
                pc = np.exp(-np.sum(pb_val, axis=1))
                pe = ma[:dim, it][np.newaxis, :] * pa
                for itD in range(dim):
                    dp[:, itD] = dp[:, itD] + 2 * c[it] * pe[:, itD] * pc

        return (p, dp) if grad else p



funHartmann6 = FunHartmann6()
