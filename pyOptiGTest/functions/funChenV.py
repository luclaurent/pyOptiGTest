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


class FunChenV(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 04/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 global minimum
        f(x1,x2)=-2000 for (x1,x2)=(7/18,13/18)
        Design space: -500<x1,x2<500
        """

        #constants
        a=1e-3
        b=0.4
        c=0.1
        d=2
        e=1.5

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluations and derivatives
        pa=xxx-b*yyy-c
        da=a**2+pa**2
        pb=d*xxx+yyy-e
        db=a**2+pb**2
        p=-a/da-a/db

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*a*pa/da**2 +2*a*d*pb/db**2
            dp[:, 1]=-2*a*b*pa/da**2 +2*a*pb/db**2

        return (p, dp) if grad else p


funChenV = FunChenV()
