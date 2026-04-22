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


class FunBukin01(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Bukin 1's function
        L. LAURENT -- 01/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        """

        #constants
        a=4
        b=100
        c=6
        d=8
        e=4

        #evaluation and derivatives
        pa=(X[:, 0]-X[:, 1])**2-a
        pb=c*(X[:, 0]**2+X[:, 1]**2)+d*X[:, 0]*X[:, 1]-e
        p=(pa**2+b*pb**2)
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=4*(X[:, 0]-X[:, 1])*pa +2*b*(2*c*X[:, 0]+d*X[:, 1])*pb
            dp[:, 1]=-4*(X[:, 0]-X[:, 1])*pa +2*b*(2*c*X[:, 1]+d*X[:, 0])*pb

        return (p, dp) if grad else p


funBukin01 = FunBukin01()
