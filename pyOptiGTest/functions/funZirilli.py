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


class FunZirilli(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Zirilli function
        L. LAURENT -- 28/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(-1.0465,0)=-0.3523
        Design space -10<xi<10
        """

        #constants
        a=1/4
        b=1/2
        c=0.1
        d=1/2

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        p=a*x**4-b*x**2+c*x+d*y**2

        if grad:
            dp=np.zeros_like(X)
            #
            dp[:, 0]=4*a*x**3-2*b*x+c
            dp[:, 1]=2*d*y



        return (p, dp) if grad else p


funZirilli = FunZirilli()
