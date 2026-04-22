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


class FunCamelbackThreeHump(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 04/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 local minimum
        f(x1,x2)=0 for (x1,x2)=(0,0)
        Design space: -5<x1<5 -5<x2<5
        """

        #constants
        a=2
        b=1.05
        c=6

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluations and derivatives
        p=a*xxx**2 -b*xxx**4 +xxx**6/c +xxx*yyy +yyy**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*a*xxx -4*b*xxx**3 +6/c*xxx**5 +yyy
            dp[:, 1]=xxx +2*yyy

        return (p, dp) if grad else p


funCamelbackThreeHump = FunCamelbackThreeHump()
