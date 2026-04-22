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


class FunTrigonometric3(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Trigonometric 3 function
        L. LAURENT -- 23/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(xx,yy)=zz
        Design space -3<xi<3
        """


        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        p=np.sin(x)+np.cos(y)

        if grad:
            #
            dp=np.zeros_like(X)
            #
            dp[:, 0]=np.cos(x)
            dp[:, 1]=-np.sin(y)

        return (p, dp) if grad else p


funTrigonometric3 = FunTrigonometric3()
