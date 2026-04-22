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


class FunBeale(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 16/09/2011 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2)=0 pour (x1,x2)=(3,0.5)
        Design space: -4.5<x1<4.5, -4.5<x<4.5
        """

        #responses and derivatives
        xxx=X[:, 0]
        yyy=X[:, 1]

        p = (1.5 - xxx + xxx*yyy)**2 + (2.25 - xxx + xxx*yyy**2)**2 + (2.625 - xxx + xxx*yyy**3)**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*(yyy-1)*(1.5-xxx+xxx*yyy)+ 2*(yyy**2-1)*(2.25 - xxx + xxx*yyy**2) + 2*(yyy**3-1)*(2.625 - xxx + xxx*yyy**3)
            dp[:, 1]=2*xxx*(1.5-xxx+xxx*yyy)+ 4*xxx*yyy*(2.25 - xxx + xxx*yyy**2) + 6*xxx*yyy**2*(2.625 - xxx + xxx*yyy**3)

        return (p, dp) if grad else p


funBeale = FunBeale()
