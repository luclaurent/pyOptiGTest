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


class FunNewFunction2(TestFunction):
    def evaluate(self, X, grad=False):
        """
        New Function 2 function
        L. LAURENT -- 21/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=-0.1971881059905 for x=[-9.94112 -9.99952]
        Design space: -10<xi<10
        """
        #constants
        a=100

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        xy=x**2+y
        axy=np.abs(xy)
        sqxy=np.sqrt(axy)
        sa=np.sin(sqxy)
        #
        pb=(x+y)/a
        #
        p=np.abs(sa)**(1/2)+pb
        #
        if grad:
            #
            dp=np.zeros_like(X)
            #
            ca=np.cos(sqxy)
            #
            dp[:, 0]=1/2*x*np.sign(xy)/sqxy*ca*np.sign(sa)*np.abs(sa)**(-1/2)+1/a
            dp[:, 1]=1/4*np.sign(xy)/sqxy*ca*np.sign(sa)*np.abs(sa)**(-1/2)+1/a
            #

        return (p, dp) if grad else p


funNewFunction2 = FunNewFunction2()
