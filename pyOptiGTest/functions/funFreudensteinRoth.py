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


class FunFreudensteinRoth(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 16/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2)=0 for (5,4)
        Design space: -10<xi<10
        """

        #constants
        a=13
        b=5
        c=2
        d=29
        e=1
        f=14

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        pa=xxx-a+yyy*((b-yyy)*yyy-c)
        pb=xxx-d+((yyy+e)*yyy-f)*yyy
        p=pa**2+pb**2
        #
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*(pa+pb)
            dp[:, 1]=2*((b-yyy)*yyy -c+yyy*(b-2*yyy))*pa +2*(yyy*(yyy+e)-f+yyy*(2*yyy+e))*pb

        return (p, dp) if grad else p


funFreudensteinRoth = FunFreudensteinRoth()
