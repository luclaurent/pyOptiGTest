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


class FunWatson(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Watson function
        1 minimum global: f(-0.0158,1.012,-0.2329,1.26,-1.513,0.9928)=0.002288
        Design space 0.25<xi<10
        """
        aa = 29
        b = 4
        c = 5
        d = lambda i: i / aa

        liJ = np.arange(0, b + 1)  # [0,1,2,3,4]
        liL = np.arange(0, c + 1)  # [0,1,2,3,4,5]

        n_samples = X.shape[0]
        n_vars = X.shape[1]

        pp = np.zeros((n_samples, aa))
        for itI in range(aa):
            di = d(itI)
            # pA = (liJ-1) * di^liJ * X[:, 0:b+1]
            # Need to handle X columns - take min(b+1, n_vars)
            nb = min(b + 1, n_vars)
            nc = min(c + 1, n_vars)
            pA_coeffs = (liJ[:nb] - 1) * di**liJ[:nb]  # shape (nb,)
            pA = pA_coeffs[np.newaxis, :] * X[:, :nb]
            pB_coeffs = di**liL[:nc]  # shape (nc,)
            pB = pB_coeffs[np.newaxis, :] * X[:, :nc]
            pp[:, itI] = np.sum(pA, axis=1) - np.sum(pB, axis=1)**2 - 1

        p = np.sum(pp**2, axis=1) + X[:, 0]**2

        if grad:
            dp = np.zeros_like(X)
            # Derivative wrt x1
            pdp = np.zeros((n_samples, aa))
            for itI in range(aa):
                di = d(itI)
                nc = min(c + 1, n_vars)
                pB_coeffs = di**liL[:nc]
                pB = pB_coeffs[np.newaxis, :] * X[:, :nc]
                pdp[:, itI] = -pp[:, itI] * (1 + 2 * np.sum(pB, axis=1))
            dp[:, 0] = 2 * X[:, 0] + 2 * np.sum(pdp, axis=1)

            # Derivatives wrt x2..xn
            for itD in range(2, n_vars + 1):
                pdp2 = np.zeros((n_samples, aa))
                for itI in range(aa):
                    di = d(itI)
                    nc = min(c + 1, n_vars)
                    pB_coeffs = di**liL[:nc]
                    pB = pB_coeffs[np.newaxis, :] * X[:, :nc]
                    pdp2[:, itI] = pp[:, itI] * di**(itD - 1) * (itD - 2 - 2 * np.sum(pB, axis=1))
                dp[:, itD - 1] = 2 * np.sum(pdp2, axis=1)

        return (p, dp) if grad else p



funWatson = FunWatson()
