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


class FunPenHolder(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        4 global minimas : f(x)=-1.037845 for x=[9.646167708023526, 9.646167671043401;
        -9.646167708023526, 9.646167671043401;
        9.646167708023526, -9.646167671043401;
        -9.646167708023526, -9.646167671043401];
        Design space: -11<xi<11
        """

        #constants
        a=1
        b=np.pi

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        sxy=a-np.sqrt(xxx**2+yyy**2)/b
        g=np.exp(np.abs(sxy))
        h=np.cos(xxx)*np.cos(yyy)
        #
        p=-np.exp(np.abs(g*h)**(-1))

        if grad:
            dgX=-1/b*np.sign(sxy)*xxx/np.sqrt(xxx**2+yyy**2)*g
            dgY=-1/b*np.sign(sxy)*yyy/np.sqrt(xxx**2+yyy**2)*g
            dhX=-np.cos(yyy)*np.sin(xxx)
            dhY=-np.sin(yyy)*np.cos(xxx)
            #
            dp = np.zeros_like(X)
            dp[:, 0]=-np.sign(g*h)*(g*h)**(-2) *(dgX*h+dhX*g)*p
            dp[:, 1]=-np.sign(g*h)*(g*h)**(-2) *(dgY*h+dhY*g)*p

        return (p, dp) if grad else p


funPenHolder = FunPenHolder()
