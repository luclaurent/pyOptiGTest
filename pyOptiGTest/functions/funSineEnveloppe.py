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


class FunSineEnveloppe(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Sine Enveloppe function
        1 global minimum: f(0,...,0)=0
        Design space: -100<xi<100
        """
        a = 0.5
        b = 1e-3
        c = 1.0
        nbvar = X.shape[1]

        xkp = X[:, 1:]
        xk = X[:, :-1]
        xy2 = xkp ** 2 + xk ** 2
        sqxy = np.sqrt(xy2)
        hval = np.sin(sqxy) ** 2 - a
        gval = (b * xy2 + c) ** 2

        pa = hval / gval + a
        p = np.sum(pa, axis=1)

        if grad:
            dp = np.zeros_like(X)
            for itX in range(nbvar):
                if itX == 0:
                    x_val = X[:, 1]
                    y_val = X[:, 0]
                    xy2_ = x_val ** 2 + y_val ** 2
                    sqxy_ = np.sqrt(xy2_)
                    h_ = np.sin(sqxy_) ** 2 - a
                    g_ = (b * xy2_ + c) ** 2
                    dhy = 2 * y_val / sqxy_ * np.cos(sqxy_) * np.sin(sqxy_)
                    dgy = 4 * b * y_val * (b * xy2_ + c)
                    dp[:, itX] = (dhy * g_ - h_ * dgy) / g_ ** 2
                elif itX == nbvar - 1:
                    x_val = X[:, nbvar - 1]
                    y_val = X[:, nbvar - 2]
                    xy2_ = x_val ** 2 + y_val ** 2
                    sqxy_ = np.sqrt(xy2_)
                    h_ = np.sin(sqxy_) ** 2 - a
                    g_ = (b * xy2_ + c) ** 2
                    dhx = 2 * x_val / sqxy_ * np.cos(sqxy_) * np.sin(sqxy_)
                    dgx = 4 * b * x_val * (b * xy2_ + c)
                    dp[:, itX] = (dhx * g_ - h_ * dgx) / g_ ** 2
                else:
                    # Contribution from pair (itX, itX-1)
                    xk_ = X[:, itX]
                    xkm = X[:, itX - 1]
                    xy2_1 = xk_ ** 2 + xkm ** 2
                    sqxy_1 = np.sqrt(xy2_1)
                    h_1 = np.sin(sqxy_1) ** 2 - a
                    g_1 = (b * xy2_1 + c) ** 2
                    dhx1 = 2 * xk_ / sqxy_1 * np.cos(sqxy_1) * np.sin(sqxy_1)
                    dgx1 = 4 * b * xk_ * (b * xy2_1 + c)
                    # Contribution from pair (itX+1, itX)
                    xkp_ = X[:, itX + 1]
                    xy2_2 = xkp_ ** 2 + xk_ ** 2
                    sqxy_2 = np.sqrt(xy2_2)
                    h_2 = np.sin(sqxy_2) ** 2 - a
                    g_2 = (b * xy2_2 + c) ** 2
                    dhy2 = 2 * xk_ / sqxy_2 * np.cos(sqxy_2) * np.sin(sqxy_2)
                    dgy2 = 4 * b * xk_ * (b * xy2_2 + c)
                    dp[:, itX] = ((dhx1 * g_1 - h_1 * dgx1) / g_1 ** 2
                                  + (dhy2 * g_2 - h_2 * dgy2) / g_2 ** 2)

        return (p, dp) if grad else p


funSineEnveloppe = FunSineEnveloppe()
