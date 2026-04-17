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


class FunDeckkersAarts(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Deckkers-Aarts function
        L. LAURENT -- 05/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        4 global minimas : f(x1,x2,x3,x4)=-24777 for {(0,-15), (0,15)}
        Design space: -20<xi<20
        """

        #constants
        a=1e5

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        pa=a*xxx**2+yyy**2
        pb=xxx**2+yyy**2
        p=pa-pb**2+1/a*pb**4
        #
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*a*xxx-4*xxx*pb+8/a*xxx*pb**3
            dp[:, 1]=2*yyy-4*yyy*pb+8/a*yyy*pb**3

        return (p, dp) if grad else p


funDeckkersAarts = FunDeckkersAarts()
