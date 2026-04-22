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


class FunParsopoulos(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Parsopoulos function
        L. LAURENT -- 17/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0 for x=(pi/2,pi)
        Design space: -5<xi<5
        """

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        cx=np.cos(xxx)
        sy=np.sin(yyy)
        p=cx**2+sy**2

        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=-2*np.cos(xxx)*np.sin(xxx)
            dp[:, 1]=2*np.cos(yyy)*np.sin(yyy)

        return (p, dp) if grad else p


funParsopoulos = FunParsopoulos()
