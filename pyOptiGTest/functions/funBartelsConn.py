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


class FunBartelsConn(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 31/10/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        one local minimum
        1 global minimum : x=(0,0) >> f(x)=1
        design space -500<xi<500
        """

        #responses and derivatives
        xxx=X[:, 0]
        yyy=X[:, 1]

        fa=xxx**2+yyy**2+xxx*yyy
        fb=np.sin(xxx)
        fc=np.cos(yyy)
        p=np.abs(fa)+np.abs(fb)+np.abs(fc)

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=np.sign(fa)*(2*xxx+yyy)+np.sign(fb)*(np.cos(xxx))
            dp[:, 1]=np.sign(fa)*(2*yyy+xxx)+np.sign(fc)*(-np.sin(yyy))

        return (p, dp) if grad else p


funBartelsConn = FunBartelsConn()
