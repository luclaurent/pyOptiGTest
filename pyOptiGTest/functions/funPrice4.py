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


class FunPrice4(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Price 4 function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0 for x={(0,0),(2,4),(1.464,?2.506)}
        Design space: -500<xi<500
        """

        #constants
        a=2
        b=6

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        g=a*xxx**3*yyy-yyy**3
        h=b*xxx-yyy**2+yyy
        #
        p=g**2+h**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=6*a*xxx**2*yyy*g+2*b*h
            dp[:, 1]=2*(a*xxx**3-3*yyy**2)*g+(1-2*yyy)*h

        return (p, dp) if grad else p


funPrice4 = FunPrice4()
