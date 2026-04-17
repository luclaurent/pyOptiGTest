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


class FunBrent(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 13/12/2010 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum :
        f(x1,x2)=0 for (x1,x2)=(-10,-10)
        Design space: -10<x1,x2<10
        """

        #constants
        a=10

        #responses and derivatives
        xxx=X[:, 0]
        yyy=X[:, 1]

        p = (xxx+a)**2+(yyy+10)**2+np.exp(-xxx**2-yyy**2)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*(xxx+a)-2*xxx*np.exp(-xxx**2-yyy**2)
            dp[:, 1]=2*(yyy+a)-2*yyy*np.exp(-xxx**2-yyy**2)

        return (p, dp) if grad else p


funBrent = FunBrent()
