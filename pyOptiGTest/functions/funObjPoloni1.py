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


class FunObjPoloni1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Poloni two-objective 1st objective function.
        f1(x) = [1 + (A1 - B1(x))^2 + (A2 - B2(x))^2]
        where A1 = 0.5*sin(1)-2*cos(1)+sin(2)-1.5*cos(2)
              A2 = 1.5*sin(1)-cos(1)+2*sin(2)-0.5*cos(2)
              B1(x) = 0.5*sin(x1)-2*cos(x1)+sin(x2)-1.5*cos(x2)
              B2(x) = 1.5*sin(x1)-cos(x1)+2*sin(x2)-0.5*cos(x2)
        Design space: -pi <= xi <= pi
        """

        x1 = X[:, 0]
        x2 = X[:, 1]

        A1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
        A2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)

        B1 = 0.5 * np.sin(x1) - 2 * np.cos(x1) + np.sin(x2) - 1.5 * np.cos(x2)
        B2 = 1.5 * np.sin(x1) - np.cos(x1) + 2 * np.sin(x2) - 0.5 * np.cos(x2)

        p = 1 + (A1 - B1)**2 + (A2 - B2)**2

        if grad:
            dB1_dx1 = 0.5 * np.cos(x1) + 2 * np.sin(x1)
            dB1_dx2 = np.cos(x2) + 1.5 * np.sin(x2)
            dB2_dx1 = 1.5 * np.cos(x1) + np.sin(x1)
            dB2_dx2 = 2 * np.cos(x2) + 0.5 * np.sin(x2)

            dp = np.zeros_like(X)
            dp[:, 0] = -2 * (A1 - B1) * dB1_dx1 - 2 * (A2 - B2) * dB2_dx1
            dp[:, 1] = -2 * (A1 - B1) * dB1_dx2 - 2 * (A2 - B2) * dB2_dx2
            return p, dp
        return p


funObjPoloni1 = FunObjPoloni1()
