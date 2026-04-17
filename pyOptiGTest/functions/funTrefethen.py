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


class FunTrefethen(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Trefethen function
        L. LAURENT -- 23/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(-0.02440307923,0.2106124261)=-3.3068686474
        Design space -5<xi<5
        """

        #constants
        a=0.25
        b=0.25
        c=50
        d=10
        e=60
        f=70
        g=80

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        sx=np.sin(c*x)
        sa=np.sin(d*x+d*y)
        sb=np.sin(e*np.exp(y))
        sc=np.sin(f*np.sin(x))
        sd=np.sin(np.sin(g*y))
        #
        p=a*x**2+b*y**2+np.exp(sx)-sa +sb+sc+sd

        if grad:
            #
            dp=np.zeros_like(X)
            #
            ca=np.cos(d*x+d*y)
            #
            dp[:, 0]=2*a*x+c*np.cos(c*x)*np.exp(sx) -d*ca+f*np.cos(x)*np.cos(f*np.sin(x))
            dp[:, 1]=2*b*y-d*ca +e*np.exp(y)*np.cos(e*np.exp(y)) +g*np.cos(g*y)*np.cos(np.sin(g*y))

        return (p, dp) if grad else p


funTrefethen = FunTrefethen()
