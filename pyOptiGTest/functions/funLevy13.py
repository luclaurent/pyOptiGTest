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


class FunLevy13(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Levy 13 function
        L. LAURENT -- 17/03/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x)=0 for xi=1
        Design space: -10<xi<10
        """
        #constants
        a=1
        b=3*np.pi
        c=2*np.pi


        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        pa=np.sin(b*y)**2+a
        pb=np.sin(c*y)**2+a
        pc=x-a
        pd=y-a
        pe=np.sin(b*x)
        #
        p=pc**2*pa+pd**2*pb+pe**2

        #
        if grad:
            #
            dp=np.zeros_like(X)
            #
            dp[:, 0]=2*pc*pa+2*b*pe*np.cos(b*x)
            dp[:, 1]=2*b*pc**2*np.sin(b*y)*np.cos(b*y) +2*pd*pb +2*c*pd**2*np.sin(c*y)*np.cos(c*y)

        return (p, dp) if grad else p


funLevy13 = FunLevy13()
