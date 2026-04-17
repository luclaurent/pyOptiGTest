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


class FunHimmelblau(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Himmelblau function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2)=0 for (x1,x2)=(3,2)
        Design space: -5<xi<5
        """
        #constants
        a=11
        b=7

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        pa=xxx**2+yyy-a
        pb=xxx+yyy**2-b
        #
        p=pa**2+pb**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=4*xxx*pa+2*pb
            dp[:, 1]=2*pa+4*yyy*pb

        return (p, dp) if grad else p


funHimmelblau = FunHimmelblau()
