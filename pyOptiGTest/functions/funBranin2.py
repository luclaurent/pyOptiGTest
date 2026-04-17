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


class FunBranin2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 13/12/2010 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        3 global minima :
        f(x1,x2)=5.559037 for (x1,x2)=(-3.2,12.53)
        Design space: -5<x1,x2<15
        """

        #constants
        a=5.1/(4*np.pi**2)
        b=5/np.pi
        c=6
        d=10
        e=1/(8*np.pi)

        #responses and derivatives
        xxx=X[:, 0]
        yyy=X[:, 1]

        p = (yyy-a*xxx**2+b*xxx-c)**2 +d*(1-e)*np.cos(xxx)*np.cos(yyy)+np.log(xxx**2+yyy**2+1)+d

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*(yyy-a*xxx**2+b*xxx-c)*(b-2*a*xxx) -d*(1-e)*np.sin(xxx)*np.cos(yyy) +2*xxx/(xxx**2+yyy**2+1)
            dp[:, 1]=2*(yyy-a*xxx**2+b*xxx-c) -d*(1-e)*np.cos(xxx)*np.sin(yyy) +2*yyy/(xxx**2+yyy**2+1)

        return (p, dp) if grad else p


funBranin2 = FunBranin2()
