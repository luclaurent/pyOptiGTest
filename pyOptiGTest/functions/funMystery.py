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


class FunMystery(TestFunction):
    def evaluate(self, X, grad=False):
        """
        "Mystery" function (Sasena 2002)
        L. LAURENT -- 01/04/2011 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        3 local minima
        1 global minimum: f(x)=-1.4565 pour x={2.5044,2.5778}
        design space: 0<xi<5
        """

        a=2
        b=0.01
        c=2
        d=2
        e=7
        f=0.5
        g=0.7

        #responses and derivatives
        xxx=X[:, 0]
        yyy=X[:, 1]

        myst=a+b*(yyy-xxx**2)**2+(1-xxx)**2+c*(d-yyy)**2+e*np.sin(f*xxx)*np.sin(g*xxx*yyy)
        p=myst
        if grad:
            dmyst=np.zeros_like(X)
            dmyst[:, 0]=-b*4*(yyy-xxx**2)*xxx-2*(1-xxx)+ e*f*np.cos(f*xxx)*np.sin(g*xxx*yyy)+e*g*yyy*np.sin(f*xxx)*np.cos(g*xxx*yyy)
            dmyst[:, 1]=2*b*(yyy-xxx**2)-4*(d-yyy)+e*g*xxx*np.sin(f*xxx)*np.cos(g*xxx*yyy)
            dp=dmyst

        return (p, dp) if grad else p


funMystery = FunMystery()
