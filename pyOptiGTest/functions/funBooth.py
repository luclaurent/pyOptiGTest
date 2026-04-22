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


class FunBooth(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 16/09/2011 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum: f(x1,x2)=0 pour (x1,x2)=(1,3)
        Design space: -10<x1<10, -10<x<10
        """

        #responses and derivatives
        xxx=X[:, 0]
        yyy=X[:, 1]

        p = (xxx+2*yyy-7)**2+(2*xxx+yyy-5)**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*(xxx+2*yyy-7)+4*(2*xxx+yyy-5)
            dp[:, 1]=4*(xxx+2*yyy-7)+2*(2*xxx+yyy-5)

        return (p, dp) if grad else p


funBooth = FunBooth()
