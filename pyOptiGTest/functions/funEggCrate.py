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


class FunEggCrate(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 15/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2)=0 for (0,0)
        Design space: -5<xi<5
        """

        #constants
        a=25

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        sx=np.sin(xxx)
        sy=np.sin(yyy)
        p=xxx**2+yyy**2+a*(sx**2+sy**2)
        #
        if grad:
            cx=np.cos(xxx)
            cy=np.cos(yyy)
            #
            dp = np.zeros_like(X)
            dp[:, 0]=2*xxx+2*a*cx*sx
            dp[:, 1]=2*yyy+2*a*cy*sy

        return (p, dp) if grad else p


funEggCrate = FunEggCrate()
