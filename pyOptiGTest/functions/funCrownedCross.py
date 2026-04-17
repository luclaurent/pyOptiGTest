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


class FunCrownedCross(TestFunction):
    def evaluate(self, X, grad=False):
        """
        CrownedCross function
        L. LAURENT -- 14/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=1e-4 for x1=0 (forall x2) or x2=0 (forall x1)
        Design space: -10<xi<10
        """
        #constants
        a=0.1
        b=100
        c=np.pi
        d=1
        e=1e-4

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        xy=np.sqrt(x**2+y**2)
        fxy=b-xy/c
        #
        pa=np.exp(np.abs(fxy))
        sx=np.sin(x)
        sy=np.sin(y)
        #
        h=pa*sx*sy
        g=np.abs(h)+d
        #
        p=e*g**a
        #
        if grad:
            #
            dh=np.cos(X)*np.sin(X[:, [1, 0]])*pa[:, None]-1/c*X/xy[:, None]*np.sign(fxy)[:, None]*h[:, None]
            #
            dg=np.sign(h)[:, None]*dh
            #
            dp=e*a*dg*g[:, None]**(a-1)

        return (p, dp) if grad else p


funCrownedCross = FunCrownedCross()
