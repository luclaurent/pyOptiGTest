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


class FunPrice3(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Price 3 function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0 for x=(1,1)
        Design space: -500<xi<500
        """

        #constants
        a=100
        b=6
        c=6.4
        d=0.6
        e=0.5

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        pa=yyy-xxx**2
        pb=c*(yyy-e)**2-xxx-d
        #
        p=a*pa**2+b*pb**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=-4*a*xxx*pa-2*b*pb
            dp[:, 1]=2*a*pa+4*b*c*(yyy-e)*pb

        return (p, dp) if grad else p


funPrice3 = FunPrice3()
