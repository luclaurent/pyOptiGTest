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


class FunElAttarVidyasogarDutta(TestFunction):
    def evaluate(self, X, grad=False):
        """
        L. LAURENT -- 15/11/2016 -- luc.laurent@lecnam.net
        sources available here:
        https://bitbucket.org/luclaurent/optigtest/
        https://github.com/luclaurent/optigtest/
        global minimum : f(x1,x2)=1.712780354 for (3.40918683, -2.17143304)
        Design space: -500<xi<500
        """

        #constants
        a=10
        b=7
        c=1

        #variables
        xxx=X[:, 0]
        yyy=X[:, 1]

        #evaluation and derivatives
        pa=xxx**2+yyy-a
        pb=xxx+yyy**2-b
        pc=xxx**2+yyy**3-c
        p=pa**2+pb**2+pc**2
        #
        if grad:
            dp = np.zeros_like(X)
            dp[:, 0]=4*xxx*pa+2*pb+4*xxx*pc
            dp[:, 1]=2*pa+4*yyy*pb+6*yyy**2*pc

        return (p, dp) if grad else p


funElAttarVidyasogarDutta = FunElAttarVidyasogarDutta()
