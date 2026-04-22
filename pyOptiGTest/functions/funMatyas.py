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


class FunMatyas(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Matyas's function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2)=0 for (x1,x2)=(0,0)
        Design space: -10<xi<10
        """
        #constants
        a=0.26
        b=0.48

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        p=a*(xxx**2+yyy**2)-b*xxx*yyy

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*a*xxx-b*yyy
            dp[:, 1]=2*a*yyy-b*xxx

        return (p, dp) if grad else p


funMatyas = FunMatyas()
