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


class FunChenBird(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Chen Bird's function
        L. LAURENT -- 04/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        2 global minima
        f(x1,x2)=-2000 for (x1,x2)={(0.5,0.5),(-0.5,-0.5)}
        Design space: -500<x1,x2<500
        """

        #constants
        a=1e-3
        b=1
        c=0.5

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluations and derivatives
        pa=xxx**2+yyy**2-b
        da=a**2+pa**2
        pb=xxx**2+yyy**2-c
        db=a**2+pb**2
        pc=xxx-yyy
        dc=a**2+pc**2
        p=-a/da-a/db-a/dc

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=4*a*xxx*pa/da**2 +4*a*xxx*pb/db**2 +2*a*pc/dc**2
            dp[:, 1]=4*a*yyy*pa/da**2 +4*a*yyy*pb/db**2 -2*a*pc/dc**2

        return (p, dp) if grad else p


funChenBird = FunChenBird()
