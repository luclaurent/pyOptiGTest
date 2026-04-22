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


class FunGoldsteinPrice(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 13/12/2010 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2)=3 for (x1,x2)=(0,-1)
        Design space: -2<x1<2, -2<x<2
        """
        #constants
        a=1
        b=19
        c=14
        d=3
        e=6
        f=3
        g=30
        h=2
        k=18
        l=32
        m=12
        n=48
        o=36
        pp=27

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        qq=xxx+yyy+a
        rr=b-c*xxx+d*xxx**2-c*yyy+e*xxx*yyy+f*yyy**2
        ss=h*xxx-f*yyy
        tt=k-l*xxx+m*xxx**2+n*yyy-o*xxx*yyy+pp*yyy**2
        #
        a = a+qq**2*rr
        b = g+ss**2*tt
        p = a*b

        if grad:
            dqqX=1
            dqqY=1
            drrX=-c+2*d*xxx+e*yyy
            drrY=-c+e*xxx+2*f*yyy
            dssX=h
            dssY=-f
            dttX=-l+2*m*xxx-o*yyy
            dttY=n-o*xxx+2*pp*yyy
            #
            dp = np.zeros_like(X)
            dp[:, 0]=(2*dqqX*qq*rr+qq**2*drrX)*b +(2*dssX*ss*tt+ss**2*dttX)*a
            dp[:, 1]=(2*dqqY*qq*rr+qq**2*drrY)*b +(2*dssY*ss*tt+ss**2*dttY)*a

        return (p, dp) if grad else p


funGoldsteinPrice = FunGoldsteinPrice()
