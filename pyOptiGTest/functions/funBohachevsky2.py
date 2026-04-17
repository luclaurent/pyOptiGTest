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


class FunBohachevsky2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bohachevsky2 function
        L. LAURENT -- 16/09/2011 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum: f(x1,x2)=0 pour (x1,x2)=(0,0)
        Design space: -100<x1<100, -100<x2<100
        """

        #responses and derivatives
        xxx=X[:, 0]
        yyy=X[:, 1]

        p = xxx**2+2*yyy**2-0.3*np.cos(3*np.pi*xxx)*np.cos(4*np.pi*yyy)+0.3

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*xxx+0.9*np.pi*np.sin(3*np.pi*xxx)*np.cos(4*np.pi*yyy)
            dp[:, 1]=4*yyy+1.2*np.pi*np.cos(3*np.pi*xxx)*np.sin(4*np.pi*yyy)

        return (p, dp) if grad else p


funBohachevsky2 = FunBohachevsky2()
