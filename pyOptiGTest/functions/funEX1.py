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


class FunEX1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        EX1 function
        L. LAURENT -- 16/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2)=-1.28186 for (1.764,11.15)
        Design space: 0<xi<12
        """

        #constants
        a=0.1
        b=1
        c=10
        d=11

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        p=a*(b-xxx**2)+a*np.sin(c*xxx)+(d-yyy)**2+np.sin(c*yyy)
        #
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=-2*xxx*a+a*c*np.cos(c*xxx)
            dp[:, 1]=-2*(d-yyy)+c*np.cos(c*yyy)

        return (p, dp) if grad else p


funEX1 = FunEX1()
