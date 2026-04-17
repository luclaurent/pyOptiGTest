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


class FunHansen(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Hansen function
        global minimum: f(x1,x2)=-176.5418
        Design space: -10<xi<10
        """
        a = 4
        b_val = 1
        c = 2

        xxx = X[:, 0]
        yyy = X[:, 1]

        ii = np.arange(0, a + 1, dtype=float)  # 0..4

        # Broadcasting: xxx is (n,), ii is (5,) -> (n, 5)
        pa = ii[np.newaxis, :] * xxx[:, np.newaxis]
        pb_val = ii[np.newaxis, :] + pa
        pc = (ii[np.newaxis, :] + c) * yyy[:, np.newaxis]
        pd = ii[np.newaxis, :] + pc
        pe = (ii[np.newaxis, :] + b_val) * np.cos(pb_val + b_val)
        pf = (ii[np.newaxis, :] + b_val) * np.cos(pd + b_val)

        pta = np.sum(pe, axis=1)
        ptb = np.sum(pf, axis=1)
        p = pta * ptb

        if grad:
            pg = ii[np.newaxis, :] * (ii[np.newaxis, :] + b_val) * np.sin(pb_val + b_val)
            ph = (ii[np.newaxis, :] + c) * (ii[np.newaxis, :] + b_val) * np.sin(pd + b_val)

            dp = np.zeros_like(X)
            dp[:, 0] = -np.sum(pg, axis=1) * ptb
            dp[:, 1] = -np.sum(ph, axis=1) * pta

        return (p, dp) if grad else p


funHansen = FunHansen()
