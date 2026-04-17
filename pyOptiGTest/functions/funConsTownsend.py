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


class FunConsTownsend(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Townsend constraint function.
        g(x) = x1^2 + x2^2 - [2*cos(t) - 0.5*cos(2t) - 0.25*cos(3t) - 0.125*cos(4t)]^2 - 4*sin(t)^2 < 0
        where t = atan2(x1, x2)
        """

        xxx = X[:, 0]
        yyy = X[:, 1]

        a = 2.0
        b = 0.5
        c = 0.25
        d = 0.125
        e = 3.0
        f = 4.0

        t = np.arctan2(xxx, yyy)
        st = np.sin(t)
        ct = np.cos(t)
        cat = np.cos(a * t)
        cet = np.cos(e * t)
        cft = np.cos(f * t)

        td = a * ct - b * cat - c * cet - d * cft
        p = xxx**2 + yyy**2 - f * st**2 - td**2

        if grad:
            dp = np.zeros_like(X)
            dtx = yyy / (xxx**2 + yyy**2)
            dty = -xxx / (xxx**2 + yyy**2)

            sat = np.sin(a * t)
            set_ = np.sin(e * t)
            sft = np.sin(f * t)

            dp[:, 0] = 2 * xxx - 2 * f * dtx * ct * st - 2 * dtx * (-a * st + sat + e * c * set_ + b * sft) * td
            dp[:, 1] = 2 * yyy - 2 * f * dty * ct * st - 2 * dty * (-a * st + sat + e * c * set_ + b * sft) * td
            return p, dp
        return p


funConsTownsend = FunConsTownsend()
