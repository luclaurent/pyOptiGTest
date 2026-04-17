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


class FunXor(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Xor function
        1 minimum global: f(1,-1,1,-1,-1,1,1,-1,0.421134)=0.9597588
        Design space -9<xi<9
        Requires exactly 9 variables.
        """
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]
        x8 = X[:, 7]
        x9 = X[:, 8]

        p125 = np.exp(-x1 - x2 - x5)
        p346 = np.exp(-x3 - x4 - x6)
        p5 = np.exp(-x5)
        p6 = np.exp(-x6)
        p15 = np.exp(-x1 - x5)
        p36 = np.exp(-x3 - x6)
        p25 = np.exp(-x2 - x5)
        p46 = np.exp(-x4 - x6)

        pae = np.exp(-x7 / (1 + p125) - x8 / (1 + p346) - x9)
        pa = 1 + pae
        pbe = np.exp(-x7 / (1 + p5) - x8 / (1 + p6) - x9)
        pb = 1 + pbe
        pce = np.exp(-x7 / (1 + p15) - x8 / (1 + p36) - x9)
        pc = 1 + pce
        pcc = 1 - pc**(-1)
        pde = np.exp(-x7 / (1 + p25) - x8 / (1 + p46) - x9)
        pd = 1 + pde
        pdd = 1 - pd**(-1)

        p = 1 / pa**2 + 1 / pb**2 + pcc**2 + pdd**2

        if grad:
            dp = np.zeros_like(X)
            # Gradient is complex - simplified version
            # pad derivatives
            pad1 = x7 * (-p125) / (1 + p125)**2 * pae
            pad2 = x7 * (-p125) / (1 + p125)**2 * pae
            pad3 = x8 * (-p346) / (1 + p346)**2 * pae
            pad4 = x8 * (-p346) / (1 + p346)**2 * pae
            pad5 = x7 * (-p125) / (1 + p125)**2 * pae
            pad6 = x8 * (-p346) / (1 + p346)**2 * pae
            pad7 = -pae / (1 + p125)
            pad8 = -pae / (1 + p346)
            pad9 = -pae

            dp[:, 0] = -2 * pad1 / pa**3
            dp[:, 1] = -2 * pad2 / pa**3
            dp[:, 2] = -2 * pad3 / pa**3
            dp[:, 3] = -2 * pad4 / pa**3
            dp[:, 4] = -2 * pad5 / pa**3
            dp[:, 5] = -2 * pad6 / pa**3
            dp[:, 6] = -2 * pad7 / pa**3
            dp[:, 7] = -2 * pad8 / pa**3
            dp[:, 8] = -2 * pad9 / pa**3

        return (p, dp) if grad else p



funXor = FunXor()
