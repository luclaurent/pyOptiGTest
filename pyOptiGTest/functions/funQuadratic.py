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


class FunQuadratic(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Quadratic function
        L. LAURENT -- 19/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=-3873.7243 for x=(0.19388,0.48513)
        Design space: -10<xi<10
        """

        #constants
        a=-3803.84
        b=-138.08
        c=-232.92
        d=128.08
        e=203.64
        f=182.25

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        p=a+b*xxx+c*yyy+d*xxx**2+e*yyy**2+f*xxx*yyy

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=b+2*d*xxx+f*yyy
            dp[:, 1]=c+2*e*yyy+f*xxx

        return (p, dp) if grad else p


funQuadratic = FunQuadratic()
