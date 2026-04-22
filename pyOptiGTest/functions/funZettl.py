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


class FunZettl(TestFunction):
    def evaluate(self, X, grad=False):
        """
        Zettl function
        L. LAURENT -- 28/02/2017 -- luc.laurent@lecnam.net
        sources available here:
        https://github.com/luclaurent/pyoptigtest/
        https://github.com/luclaurent/optigtest/
        1 minimum global: f(-0.02896,0)=-0.0037912
        Design space -1<xi<5
        """

        #constants
        a=1/4
        b=2

        #evaluation and derivatives
        x=X[:, 0]
        y=X[:, 1]
        #
        pa=a*x
        pb=x**2-b*x+y**2
        #
        p=pa+pb**2

        if grad:
            dp=np.zeros_like(X)
            #
            dp[:, 0]=a+2*(2*x-b)*pb
            dp[:, 1]=4*y*pb

        return (p, dp) if grad else p


funZettl = FunZettl()
