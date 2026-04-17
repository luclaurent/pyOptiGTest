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


class FunEasom(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 15/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        4 global minimas : f(x1,x2)=-1 for (pi,pi)
        Design space: -100<xi<100
        """

        #constants
        a=np.pi

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        cx=np.cos(xxx)
        cy=np.cos(yyy)
        xp=xxx-a
        yp=yyy-a
        exy=np.exp(-xp**2-yp**2)
        p=-cx*cy*exy
        #
        if grad:
            sx=np.sin(xxx)
            sy=np.sin(yyy)
            #
            dp = np.zeros_like(X)
            dp[:, 0]=exy*(sx*cy+2*xp*cx*cy)
            dp[:, 1]=exy*(sy*cx+2*yp*cx*cy)

        return (p, dp) if grad else p


funEasom = FunEasom()
