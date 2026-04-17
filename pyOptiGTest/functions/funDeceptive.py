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


class FunDeceptive(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Deceptive function
        global minimum: f(x)=-1 for xi=alphai=i/(dim+1)
        Design space: 0<xi<1
        """
        beta = 2
        a = 4.0 / 5.0
        b = 5.0
        c = 4.0
        d = 1.0

        nbvar = X.shape[1]
        n_samples = X.shape[0]
        alpha = np.arange(1, nbvar + 1, dtype=float) / (nbvar + 1)

        g = np.zeros_like(X)

        ua = a * alpha[np.newaxis, :]
        IXa = (X >= 0) & (X < ua)
        IXb = (X >= ua) & (X < alpha[np.newaxis, :])
        uc = 1.0 / b + a * alpha[np.newaxis, :]
        IXc = (X >= alpha[np.newaxis, :]) & (X < uc)
        IXd = (X >= uc) & (X <= d)

        g[IXa] = a - X[IXa] / np.broadcast_to(alpha, X.shape)[IXa]
        g[IXb] = b * X[IXb] / np.broadcast_to(alpha, X.shape)[IXb] - c
        alpha_bc = np.broadcast_to(alpha, X.shape)
        g[IXc] = b * (X[IXc] - alpha_bc[IXc]) / (alpha_bc[IXc] - d) + d
        g[IXd] = (X[IXd] - d) / (d - alpha_bc[IXd]) + a

        pa = np.sum(g, axis=1)
        p = -(1.0 / nbvar * pa) ** beta

        if grad:
            dg = np.zeros_like(X)
            dg[IXa] = -1.0 / alpha_bc[IXa]
            dg[IXb] = b / alpha_bc[IXb]
            dg[IXc] = b / (alpha_bc[IXc] - d)
            dg[IXd] = 1.0 / (d - alpha_bc[IXd])

            dp = -beta / nbvar * dg * (1.0 / nbvar) * pa[:, np.newaxis]

        return (p, dp) if grad else p


funDeceptive = FunDeceptive()
