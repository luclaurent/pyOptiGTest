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


class FunHosaki(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Hosaki function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2)=-2.3458 for (x1,x2)=(4,2)
        Design space: 0<xi<5
        """
        #constants
        a=1
        b=8
        c=7
        d=7/3
        e=1/4

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        pa=a-b*xxx+c*xxx**2-d*xxx**3+e*xxx**4
        pb=np.exp(-yyy)
        #
        p=pa*yyy**2*pb

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=(-b+2*c*xxx-3*d*xxx**2+4*e*xxx**3)*yyy**2*pb
            dp[:, 1]=pa*pb*(2*yyy-yyy**2)

        return (p, dp) if grad else p


funHosaki = FunHosaki()
