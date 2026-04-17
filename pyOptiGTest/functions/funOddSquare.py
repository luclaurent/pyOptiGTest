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


class FunOddSquare(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Odd Square function
        Design space: -5*pi<xi<5*pi
        """
        a = 2 * np.pi
        b = np.pi
        e = 0.02
        f = 0.01
        BV_full = np.array([1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4,
                            1, 1.3, 0.8, -0.4, -1.3, 1.6, -0.2, -0.6, 0.5, 1.4])

        nbvar = X.shape[1]
        n_samples = X.shape[0]
        BV = BV_full[:nbvar]

        xb = X - BV[np.newaxis, :]

        vX = np.max(xb ** 2, axis=1)
        IXm = np.argmax(xb ** 2, axis=1)
        maxD = nbvar * vX
        h = np.sum(xb ** 2, axis=1)

        pa = np.exp(-maxD / a)
        pb_val = np.cos(b * maxD)
        pc = 1.0 + e * h / (maxD + f)

        p = -pa * pb_val * pc

        if grad:
            dp = np.zeros_like(X)
            dD = np.zeros_like(X)
            for i in range(n_samples):
                dD[i, IXm[i]] = 2 * nbvar * xb[i, IXm[i]]

            dh = 2 * xb

            dp = (1.0 / a * dD * pa[:, np.newaxis] * pb_val[:, np.newaxis] * pc[:, np.newaxis]
                  + b * dD * np.sin(b * maxD)[:, np.newaxis] * pa[:, np.newaxis] * pc[:, np.newaxis]
                  - e * (dh * (maxD + f)[:, np.newaxis] - h[:, np.newaxis] * dD) / (maxD + f)[:, np.newaxis] ** 2
                  * pa[:, np.newaxis] * pb_val[:, np.newaxis])

        return (p, dp) if grad else p


funOddSquare = FunOddSquare()
