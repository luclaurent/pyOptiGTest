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


class FunSlope(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Slope function
        L. LAURENT -- 15/12/2011 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        """

        #direction of the slope
        dir=2

        #constant
        val=10

        p=val*X[:, dir-1]
        if grad:
            dp=np.zeros_like(X)
            dp[:, dir-1]=val*np.ones_like(p)

        return (p, dp) if grad else p


funSlope = FunSlope()
