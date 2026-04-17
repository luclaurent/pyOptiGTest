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


class FunTubeHolder(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 22/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(pi/2,0)=-10.872299901558
        Design space -10<xi<10
        """

        #constants
        a=4
        b=1/200

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        pa=b*x**2+b*y**2
        ca=np.cos(pa)
        sx=np.sin(x)
        cy=np.cos(y)
        #
        g=np.exp(np.abs(ca))*sx*cy
        p=-a*np.abs(g)

        if grad:
            #
            cx=np.cos(x)
            sy=np.sin(y)
            dk=np.zeros_like(X)
            dk[:, 0]=cx*cy
            dk[:, 1]=-sx*sy
            #
            sa=np.sin(pa)
            dh=-2*b*X*sa[:, None]*np.sign(ca)[:, None]
            #
            dg=dh*g[:, None]+np.exp(np.abs(ca))[:, None]*dk
            #
            dp=-a*dg*np.sign(g)[:, None]

        return (p, dp) if grad else p


funTubeHolder = FunTubeHolder()
