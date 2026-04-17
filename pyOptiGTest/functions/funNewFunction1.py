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


class FunNewFunction1(TestFunction):
    def evaluate(self, X, grad=False):
        """
        New Function 1 function
        L. LAURENT -- 20/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=-0.17894509347721144 for x=[-8.4666,-9.9988]
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
        ca=np.cos(sqxy)
        #
        pb=(x+y)/a
        #
        p=np.abs(ca)**(1/2)+pb
        #
        if grad:
            #
            dp=np.zeros_like(X)
            #
            sa=np.sin(sqxy)
            #
            dp[:, 0]=-1/2*x*np.sign(xy)/sqxy*sa*np.sign(ca)*np.abs(ca)**(-1/2)+1/a
            dp[:, 1]=-1/4*np.sign(xy)/sqxy*sa*np.sign(ca)*np.abs(ca)**(-1/2)+1/a
            #

        return (p, dp) if grad else p


funNewFunction1 = FunNewFunction1()
