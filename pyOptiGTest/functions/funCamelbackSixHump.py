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


class FunCamelbackSixHump(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Six-Hump camel back function
        L. LAURENT -- 13/12/2010 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        6 local minima and  2 global:
        f(x1,x2)=-1.0316 for (x1,x2)={(-0.0898,0.7126),(0.0898,0.7126)}
        Design space: -3<x1<3 -2<x2<2
        (recommanded: -2<x1<2 -1<x2<1)
        """

        #constants
        a=4
        b=2.1
        c=3

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluations and derivatives
        pa=(a-b*xxx**2+xxx**4/c)
        pb=(-1+yyy**2)
        p=pa*xxx**2+ xxx*yyy +a*pb*yyy**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*xxx*pa +xxx**2*(-2*b*xxx+a*xxx**3/c) +yyy
            dp[:, 1]=xxx +2*a*yyy*pb +2*a*yyy**3

        return (p, dp) if grad else p


funCamelbackSixHump = FunCamelbackSixHump()
