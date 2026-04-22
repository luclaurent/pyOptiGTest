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


class FunAdjiman(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Adjiman's function
        L. LAURENT -- 31/10/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        one local minimum
        1 global minimum : x=(2,0.10578) >> f(x)=-2.02181
        design space -1<x1<2, -2<x2<2
        """

        #responses and derivatives
        xxx=X[:, 0]
        yyy=X[:, 1]

        #
        cx=np.cos(xxx)
        sy=np.sin(yyy)
        hxy=xxx/(yyy**2+1)
        p=cx*sy-hxy
        #
        if grad:
            sx=np.sin(xxx)
            cy=np.cos(yyy)
            dp = np.zeros_like(X)
            dp[:, 0]=-sx*sy-1/(yyy**2+1)
            dp[:, 1]=cx*cy+2*xxx*yyy/(yyy**2+1)**2

        return (p, dp) if grad else p


funAdjiman = FunAdjiman()
