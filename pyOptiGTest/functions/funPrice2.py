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


class FunPrice2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Price 2 function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0.9 for x=(0,0)
        Design space: -10<xi<10
        """

        #constants
        a=1
        b=0.1

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        sx=np.sin(xxx)
        sy=np.sin(yyy)
        ex=np.exp(-xxx**2-yyy**2)
        #
        p=a+sx**2+sy**2-b*ex

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=2*sx*np.cos(xxx)+2*b*xxx*ex
            dp[:, 1]=2*sy*np.cos(yyy)+2*b*yyy*ex

        return (p, dp) if grad else p


funPrice2 = FunPrice2()
