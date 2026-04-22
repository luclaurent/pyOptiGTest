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


class FunUrsem01(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Ursem 1 function
        L. LAURENT -- 24/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(1.69714,0)=-4.8168
        Design space -2.5<x1<3 -2<x<2
        """

        #constants
        a=2
        b=np.pi/2
        c=3
        d=1/2

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        pa=-np.sin(a*x-b)
        pb=-c*np.cos(y)
        #
        p=pa+pb-d*x

        if grad:
            #
            dp=np.zeros_like(X)
            #
            dp[:, 0]=-a*np.cos(a*x-b)-d
            dp[:, 1]=c*np.sin(y)

        return (p, dp) if grad else p


funUrsem01 = FunUrsem01()
